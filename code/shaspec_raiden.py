"""
ShaSpec (CVPR'23) 分类版忠实复现 —— 作为 v2 的公平 baseline。
共享编码器 r_m + 模态专属 s_m + 残差组合 f_m=r_m+MLP([r_m,s_m]) + concat 分类。
辅助任务：DCO(specific->模态id CE) + DAO(shared->对齐uniform)。
复用 train_v2 的数据与模态编码器，只替换融合机制。
"""
import sys, os, argparse, warnings, statistics as st
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from train_v2_raiden import prepare, TimeEnc, ImgEnc, TxtEnc, DEV

MODS = ["time", "image", "text"]


class ShaSpec(nn.Module):
    def __init__(s, C, dtxt, nC, D=256):
        super().__init__()
        s.enc = nn.ModuleDict({"time": TimeEnc(C), "image": ImgEnc(), "text": TxtEnc(dtxt)})
        # 每模态投到公共维 D
        s.proj = nn.ModuleDict({m: nn.Linear(s.enc[m].out, D) for m in MODS})
        s.shared = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))  # 共享
        s.spec = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ReLU(),
                                                 nn.Linear(D, D)) for m in MODS})              # 专属
        s.compos = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, D))  # 残差组合(共享)
        s.head = nn.Sequential(nn.LayerNorm(3 * D), nn.Linear(3 * D, 128), nn.ReLU(),
                               nn.Dropout(0.3), nn.Linear(128, nC))
        s.dco = nn.Linear(D, 3)  # 域分类器(specific -> 模态id)
        s.dao = nn.Linear(D, 3)  # 对齐分类器(shared -> 应无法判别)

    def forward(s, b):
        e = {m: s.proj[m](s.enc[m](b[m])) for m in MODS}
        r = {m: s.shared(e[m]) for m in MODS}          # shared
        sp = {m: s.spec[m](e[m]) for m in MODS}        # specific
        f = {m: r[m] + s.compos(torch.cat([r[m], sp[m]], -1)) for m in MODS}
        logits = s.head(torch.cat([f["time"], f["image"], f["text"]], -1))
        return logits, r, sp


def run(D, seed, epochs=60, bs=128, patience=12, alpha=0.1, beta=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    sp = D["sp"]; nC = int(D["nC"])
    tr = np.where(np.isin(sp, ["train", "training"]))[0]
    va = np.where(sp == "val")[0]; te = np.where(sp == "test")[0]
    if len(va) == 0:
        r = np.random.RandomState(seed); r.shuffle(tr); c = int(.85 * len(tr)); va, tr = tr[c:], tr[:c]
    y = torch.tensor(D["y"]); times = torch.tensor(D["times"])
    imgs = torch.tensor(D["imgs"]).float() / 255.; txt = torch.tensor(D["Xtxt"])
    C = D["times"].shape[-1]
    cw = np.bincount(D["y"][tr], minlength=nC).astype(np.float32); cw = cw.sum() / (cw + 1e-6); cw /= cw.mean()
    net = ShaSpec(C, D["Xtxt"].shape[1], nC).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))

    def bat(ix): return {"time": times[ix].to(DEV), "image": imgs[ix].to(DEV), "text": txt[ix].to(DEV)}, y[ix].to(DEV)

    def modlabel(n, m): return torch.full((n,), MODS.index(m), dtype=torch.long, device=DEV)

    def ev(ix):
        net.eval(); ps, pr = [], []
        with torch.no_grad():
            for k in range(0, len(ix), 512):
                b, _ = bat(ix[k:k + 512]); lo, _, _ = net(b)
                ps.append(lo.argmax(-1).cpu().numpy()); pr.append(torch.softmax(lo, -1).cpu().numpy())
        p = np.concatenate(ps); pr = np.concatenate(pr); yy = D["y"][ix]
        try: auc = roc_auc_score(yy, pr[:, 1]) if nC == 2 else roc_auc_score(yy, pr, multi_class="ovr")
        except Exception: auc = float("nan")
        return f1_score(yy, p, average="macro", zero_division=0), accuracy_score(yy, p), auc

    best, bstate, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); rr = np.random.RandomState(ep); o = tr.copy(); rr.shuffle(o)
        for k in range(0, len(o), bs):
            b, yb = bat(o[k:k + bs]); n = len(yb)
            lo, r, spf = net(b)
            main = lf(lo, yb)
            dco = sum(F.cross_entropy(net.dco(spf[m]), modlabel(n, m)) for m in MODS) / 3
            # DAO: shared 应无法判别模态 -> 推向 uniform
            dao = sum(F.kl_div(F.log_softmax(net.dao(r[m]), -1),
                               torch.full((n, 3), 1/3, device=DEV), reduction="batchmean") for m in MODS) / 3
            loss = main + beta * dco + alpha * dao
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        vf = ev(va)[0]
        if vf > best: best, bstate, bad = vf, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(bstate)
    return ev(te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    a = ap.parse_args()
    D = prepare(a.dataset)
    print(f"[ShaSpec {a.dataset}] n={len(D['y'])} nC={int(D['nC'])} test={(D['sp']=='test').sum()}", flush=True)
    fs = []
    for sd in a.seeds:
        f, ac, au = run(D, sd)
        fs.append(f)
        print(f"  seed{sd}: macroF1={f:.3f} acc={ac:.3f} auc={au:.3f}", flush=True)
    print(f"  >> ShaSpec macroF1 {st.mean(fs):.3f}±{(st.pstdev(fs) if len(fs)>1 else 0):.3f}", flush=True)
