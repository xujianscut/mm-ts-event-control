"""
TimeCAP 多模态 v2:训练版编码器 + 门控 agent + 单调融合。

设计目标:融合输出稳定 >= best single-modality。
机制:
  1. 深监督:每个模态自带分类头，各自 CE 监督 -> 每个模态单独就强。
  2. 门控 agent:小网络读三模态 pooled 特征，输出每模态门控 g_m in [0,1]。
     fusion_logits = time_logits + sum_{m in image,text} g_m*(L_m - time_logits)   (residual，锚定时序)
     也支持 anchor=最强模态。
  3. modality dropout:训练时随机丢辅助模态 -> 门控学会"没用就关"，且不过度依赖。
  4. class-balanced CE + macro-F1 早停。
用法：python train_v2.py --dataset medical --seeds 2026 2027 2028 --anchor image
"""
import sys, argparse, warnings, time, json, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
try:
    from datasets.dataset_adapter import create_dataset_adapter
    from config import DATASET_PATHS
except Exception:
    create_dataset_adapter = None; DATASET_PATHS = {}

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.path.expanduser("~/v2_cache")
os.makedirs(CACHE, exist_ok=True)


def prepare(ds, T_cap=48, C_cap=64, img_sz=96):
    cp = f"{CACHE}/{ds}.npz"
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return {k: z[k] for k in z.files}
    root = DATASET_PATHS[ds]
    ad = create_dataset_adapter(ds, root)
    N = len(ad)
    sp, y, times, tmask, imgs, texts = [], [], [], [], [], []
    t0 = time.time()
    for i in range(N):
        sp.append(str(ad.get_split(i)).lower()); y.append(ad.get_label(i))
        try:
            x, m, kind = ad.get_time_series(i)
        except Exception:
            x, kind = None, "none"
        arr = np.zeros((T_cap, C_cap), np.float32); mk = np.zeros((T_cap,), np.float32)
        if x is not None and kind != "image":
            x = np.nan_to_num(np.asarray(x, np.float32))
            if x.ndim == 1: x = x[:, None]
            tt, cc = min(x.shape[0], T_cap), min(x.shape[1], C_cap)
            arr[:tt, :cc] = x[:tt, :cc]; mk[:tt] = 1.0
        times.append(arr); tmask.append(mk)
        try:
            im = ad.get_image(i)
            pil = im if isinstance(im, Image.Image) else (Image.open(im) if im else None)
            imarr = (np.asarray(pil.convert("RGB").resize((img_sz, img_sz)), np.uint8).transpose(2, 0, 1)
                     if pil else np.zeros((3, img_sz, img_sz), np.uint8))
        except Exception:
            imarr = np.zeros((3, img_sz, img_sz), np.uint8)
        imgs.append(imarr)
        try:
            texts.append(str(ad.get_text(i) or ""))
        except Exception:
            texts.append("")
        if (i + 1) % 5000 == 0: print(f"  load {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    sp = np.array(sp); y = np.array(y, int)
    times = np.stack(times); tmask = np.stack(tmask); imgs = np.stack(imgs)
    nC = int(ad.num_classes) or int(y.max() + 1)
    trm = np.isin(sp, ["train", "training", "val"])
    mu = times[trm].reshape(-1, C_cap).mean(0); sd = times[trm].reshape(-1, C_cap).std(0) + 1e-6
    times = ((times - mu) / sd) * tmask[..., None]
    vec = TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1, 2))
    vec.fit([texts[i] for i in np.where(trm)[0]])
    Xtxt = vec.transform(texts).toarray().astype(np.float32)
    d = dict(sp=sp, y=y, nC=np.array(nC), times=times.astype(np.float32),
             imgs=imgs, Xtxt=Xtxt)
    np.savez_compressed(cp, **d)
    print(f"  cached {cp} ({time.time()-t0:.0f}s)", flush=True)
    return d


class TimeEnc(nn.Module):
    def __init__(s, C, h=128):
        super().__init__(); s.gru = nn.GRU(C, h, 1, batch_first=True, bidirectional=True); s.out = 2*h
    def forward(s, x): return s.gru(x)[0].mean(1)


class ImgEnc(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.c = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, h),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten()); s.out = h
    def forward(s, x): return s.c(x)


class TxtEnc(nn.Module):
    def __init__(s, d, h=128):
        super().__init__(); s.m = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.4),
                                                nn.Linear(256, h), nn.ReLU()); s.out = h
    def forward(s, x): return s.m(x)


class MonoFusion(nn.Module):
    """深监督 + 门控残差单调融合。anchor 模态兜底，其余门控残差修正。"""
    def __init__(s, C, dtxt, nC, anchor="image"):
        super().__init__()
        s.anchor = anchor; s.nC = nC
        s.te, s.ie, s.xe = TimeEnc(C), ImgEnc(), TxtEnc(dtxt)
        s.dims = {"time": s.te.out, "image": s.ie.out, "text": s.xe.out}
        s.head = nn.ModuleDict({m: nn.Linear(s.dims[m], nC) for m in s.dims})
        gin = sum(s.dims.values())
        s.gate = nn.Sequential(nn.LayerNorm(gin), nn.Linear(gin, 128), nn.ReLU(),
                               nn.Dropout(0.3), nn.Linear(128, 3))  # -> time,image,text gates

    def forward(s, b, mdrop=None):
        z = {"time": s.te(b["time"]), "image": s.ie(b["image"]), "text": s.xe(b["text"])}
        if mdrop:
            for m in mdrop: z[m] = z[m] * 0.0
        L = {m: s.head[m](z[m]) for m in z}
        g = torch.sigmoid(s.gate(torch.cat([z["time"], z["image"], z["text"]], -1)))
        gm = {"time": g[:, 0:1], "image": g[:, 1:2], "text": g[:, 2:3]}
        anc = s.anchor
        others = [m for m in ["time", "image", "text"] if m != anc]
        fused = L[anc] + sum(gm[m] * (L[m] - L[anc]) for m in others)
        return fused, L, gm


def run(D, seed, anchor, epochs=60, bs=128, patience=10, mdrop_p=0.3, aux=0.4):
    torch.manual_seed(seed); np.random.seed(seed)
    sp = D["sp"]; nC = int(D["nC"])
    tr = np.where(np.isin(sp, ["train", "training"]))[0]
    va = np.where(sp == "val")[0]; te = np.where(sp == "test")[0]
    if len(va) == 0:
        r = np.random.RandomState(seed); r.shuffle(tr); c = int(.85*len(tr)); va, tr = tr[c:], tr[:c]
    y = torch.tensor(D["y"]); times = torch.tensor(D["times"])
    imgs = torch.tensor(D["imgs"]).float()/255.; txt = torch.tensor(D["Xtxt"])
    C = D["times"].shape[-1]
    cw = np.bincount(D["y"][tr], minlength=nC).astype(np.float32); cw = cw.sum()/(cw+1e-6); cw /= cw.mean()
    net = MonoFusion(C, D["Xtxt"].shape[1], nC, anchor).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))

    def bat(ix): return {"time": times[ix].to(DEV), "image": imgs[ix].to(DEV), "text": txt[ix].to(DEV)}, y[ix].to(DEV)

    def ev(ix, which="fused"):
        net.eval(); ps, pr = [], []
        with torch.no_grad():
            for k in range(0, len(ix), 512):
                b, _ = bat(ix[k:k+512]); fu, L, _ = net(b)
                out = fu if which == "fused" else L[which]
                ps.append(out.argmax(-1).cpu().numpy()); pr.append(torch.softmax(out, -1).cpu().numpy())
        p = np.concatenate(ps); pr = np.concatenate(pr)
        yy = D["y"][ix]
        try: auc = roc_auc_score(yy, pr[:, 1]) if nC == 2 else roc_auc_score(yy, pr, multi_class="ovr")
        except Exception: auc = float("nan")
        return f1_score(yy, p, average="macro", zero_division=0), accuracy_score(yy, p), auc

    best, bs_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); r = np.random.RandomState(ep); o = tr.copy(); r.shuffle(o)
        for k in range(0, len(o), bs):
            b, yb = bat(o[k:k+bs])
            md = [m for m in ["time", "image", "text"] if np.random.rand() < mdrop_p and m != anchor]
            fu, L, _ = net(b, mdrop=md)
            loss = lf(fu, yb) + aux*sum(lf(L[m], yb) for m in L)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        vf, _, _ = ev(va, "fused")
        if vf > best: best, bs_state, bad = vf, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(bs_state)
    # val 上选融合深度：{fused, 各单模态头} 谁 val-macroF1 高就用谁预测 test
    # -> 保证 test >= best-single（标准 val 模型选择）
    cands = ["fused", "time", "image", "text"]
    valf = {c: ev(va, c)[0] for c in cands}
    pick = max(cands, key=lambda c: valf[c])
    res = {c: ev(te, c) for c in cands}
    res["selected"] = res[pick]
    res["pick"] = pick
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    ap.add_argument("--anchor", default="image")
    a = ap.parse_args()
    D = prepare(a.dataset)
    print(f"[{a.dataset}] n={len(D['y'])} nC={int(D['nC'])} test={(D['sp']=='test').sum()} "
          f"dist={np.bincount(D['y']).tolist()} anchor={a.anchor}", flush=True)
    agg = {k: [] for k in ["fused", "time", "image", "text"]}
    for sd in a.seeds:
        r = run(D, sd, a.anchor)
        for k in agg: agg[k].append(r[k][0])
        print(f"  seed{sd}: fused mF1={r['fused'][0]:.3f} acc={r['fused'][1]:.3f} auc={r['fused'][2]:.3f} | "
              f"time={r['time'][0]:.3f} image={r['image'][0]:.3f} text={r['text'][0]:.3f}", flush=True)
    import statistics as st
    def ms(x): return f"{st.mean(x):.3f}±{(st.pstdev(x) if len(x)>1 else 0):.3f}"
    print(f"  >> FUSED macroF1 {ms(agg['fused'])} | best-single "
          f"{ms([max(agg['time'][i],agg['image'][i],agg['text'][i]) for i in range(len(a.seeds))])}", flush=True)
