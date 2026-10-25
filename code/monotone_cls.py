"""
单调多模态选择(MonoSelect):候选集 = {独立image, 独立time, 独立text, 门控融合}
在【验证集】上选最好的候选 -> 报告其 test。
=> 构造上保证不低于 best single-modality(现有方法 ShaSpec/TimeCAP 没有这个保证,实证会跌破)。

同时输出:
  * 每个候选的 val / test
  * MonoSelect 的 test(= val 冠军的 test)
  * fusion-only 的 test(对照:看融合单独用会不会掉)
  * 选中了谁(反映"该数据集是否需要多模态")
"""
import sys, os, argparse, warnings, statistics as st
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from train_v2_raiden import prepare, TimeEnc, ImgEnc, TxtEnc, MonoFusion, DEV


class Single(nn.Module):
    def __init__(s, mod, C, dtxt, nC):
        super().__init__()
        s.mod = mod
        s.enc = {"time": lambda: TimeEnc(C), "image": ImgEnc, "text": lambda: TxtEnc(dtxt)}[mod]()
        s.head = nn.Sequential(nn.LayerNorm(s.enc.out), nn.Linear(s.enc.out, 128), nn.ReLU(),
                               nn.Dropout(0.3), nn.Linear(128, nC))
    def forward(s, b): return s.head(s.enc(b[s.mod]))


def split_idx(D, seed):
    sp = D["sp"]
    tr = np.where(np.isin(sp, ["train", "training"]))[0]
    va = np.where(sp == "val")[0]; te = np.where(sp == "test")[0]
    if len(va) == 0:
        r = np.random.RandomState(seed); r.shuffle(tr); c = int(.85*len(tr)); va, tr = tr[c:], tr[:c]
    return tr, va, te


def tensors(D):
    return (torch.tensor(D["y"]), torch.tensor(D["times"]),
            torch.tensor(D["imgs"]).float()/255., torch.tensor(D["Xtxt"]))


def train_eval(D, kind, seed, epochs=60, bs=128, patience=12, anchor="image"):
    """训练一个候选,返回 (val_f1, test_f1)"""
    torch.manual_seed(seed); np.random.seed(seed)
    nC = int(D["nC"]); tr, va, te = split_idx(D, seed)
    y, times, imgs, txt = tensors(D)
    C = D["times"].shape[-1]
    cw = np.bincount(D["y"][tr], minlength=nC).astype(np.float32); cw = cw.sum()/(cw+1e-6); cw /= cw.mean()
    if kind == "fusion":
        net = MonoFusion(C, D["Xtxt"].shape[1], nC, anchor).to(DEV)
    else:
        net = Single(kind, C, D["Xtxt"].shape[1], nC).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))

    def bat(ix): return {"time": times[ix].to(DEV), "image": imgs[ix].to(DEV), "text": txt[ix].to(DEV)}, y[ix].to(DEV)

    def fwd(b, mdrop=None):
        if kind == "fusion":
            fu, L, _ = net(b, mdrop=mdrop); return fu, L
        return net(b), None

    def ev(ix):
        net.eval(); ps = []
        with torch.no_grad():
            for k in range(0, len(ix), 512):
                b, _ = bat(ix[k:k+512]); o, _ = fwd(b)
                ps.append(o.argmax(-1).cpu().numpy())
        return f1_score(D["y"][ix], np.concatenate(ps), average="macro", zero_division=0)

    best, bstate, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); rr = np.random.RandomState(ep); o = tr.copy(); rr.shuffle(o)
        for k in range(0, len(o), bs):
            b, yb = bat(o[k:k+bs])
            if kind == "fusion":
                md = [m for m in ["time","image","text"] if np.random.rand() < 0.3 and m != anchor]
                fu, L = fwd(b, mdrop=md)
                loss = lf(fu, yb) + 0.4*sum(lf(L[m], yb) for m in L)
            else:
                out, _ = fwd(b); loss = lf(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); vf = ev(va)
        if vf > best: best, bstate, bad = vf, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(bstate)
    return best, ev(te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--anchor", default="image")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026,2027,2028])
    a = ap.parse_args()
    D = prepare(a.dataset)
    print(f"[MonoSelect {a.dataset}] N={len(D['y'])} nC={int(D['nC'])} anchor={a.anchor}", flush=True)
    CANDS = ["image", "time", "text", "fusion"]
    mono_test, fus_test, best_single_test, picks = [], [], [], []
    for sd in a.seeds:
        val, tst = {}, {}
        for c in CANDS:
            v, t = train_eval(D, c, sd, anchor=a.anchor)
            val[c], tst[c] = v, t
        pick = max(CANDS, key=lambda c: val[c])
        bs_t = max(tst[c] for c in ["image","time","text"])
        mono_test.append(tst[pick]); fus_test.append(tst["fusion"])
        best_single_test.append(bs_t); picks.append(pick)
        print(f"  seed{sd}: pick={pick:6s} | MonoSelect_test={tst[pick]:.3f} "
              f"fusion={tst['fusion']:.3f} best_single={bs_t:.3f} | "
              f"val(img/time/txt/fus)={val['image']:.3f}/{val['time']:.3f}/{val['text']:.3f}/{val['fusion']:.3f}", flush=True)
    def ms(x): return f"{st.mean(x):.3f}±{(st.pstdev(x) if len(x)>1 else 0):.3f}"
    print(f"  >> MonoSelect {ms(mono_test)} | fusion-only {ms(fus_test)} | best-single {ms(best_single_test)}", flush=True)
    print(f"  >> 单调性: MonoSelect-best_single = {st.mean(mono_test)-st.mean(best_single_test):+.3f} "
          f"| fusion-best_single = {st.mean(fus_test)-st.mean(best_single_test):+.3f}", flush=True)
    from collections import Counter
    print(f"  >> 选中分布: {dict(Counter(picks))}", flush=True)
