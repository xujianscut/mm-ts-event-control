"""
MIMIC-Eye 5-fold 交叉验证(按患者分组,每折内重拟合 TF-IDF/标准化 -> 严格无泄漏)
每折: test=fold_i, val=fold_{i+1}, train=其余3折
评估全部 7 个模态子集 + val 子集选择。所有折/seed 全部报告,不丢弃。
"""
import sys, os, argparse, warnings, itertools, statistics as st
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np, torch, torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from train_v2_raiden import TimeEnc, ImgEnc, TxtEnc, DEV
from subset_cls import SubsetNet

MODS = ["time","image","text"]
RAW = os.path.expanduser("~/v2_cache/mimiceye_raw.npz")

def fold_data(z, folds, i):
    pids = z["pids"].astype(str)
    te_p = folds[i]; va_p = folds[(i+1) % len(folds)]
    tr_p = set().union(*[folds[j] for j in range(len(folds)) if j not in (i,(i+1)%len(folds))])
    tr = np.where(np.isin(pids, list(tr_p)))[0]
    va = np.where(np.isin(pids, list(va_p)))[0]
    te = np.where(np.isin(pids, list(te_p)))[0]
    texts = [str(t) for t in z["texts"]]
    vec = TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1,2), stop_words="english")
    vec.fit([texts[i2] for i2 in tr])
    Xtxt = vec.transform(texts).toarray().astype(np.float32)
    times = z["times"].astype(np.float32).copy(); tmask = z["tmask"]
    mu = times[tr].reshape(-1, times.shape[-1]).mean(0); sd = times[tr].reshape(-1, times.shape[-1]).std(0)+1e-6
    times = ((times-mu)/sd) * tmask[...,None]
    imgs = z["imgs"]
    if imgs.shape[1]==1: imgs = np.repeat(imgs,3,axis=1)
    return dict(times=times, imgs=imgs, Xtxt=Xtxt, y=z["y"], nC=int(z["nC"])), tr, va, te

def run(D, tr, va, te, mods, seed, epochs=60, bs=128, patience=12):
    torch.manual_seed(seed); np.random.seed(seed)
    nC=D["nC"]; y=torch.tensor(D["y"]); times=torch.tensor(D["times"])
    imgs=torch.tensor(D["imgs"]).float()/255.; txt=torch.tensor(D["Xtxt"])
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=SubsetNet(mods, D["times"].shape[-1], D["Xtxt"].shape[1], nC).to(DEV)
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix): return {"time":times[ix].to(DEV),"image":imgs[ix].to(DEV),"text":txt[ix].to(DEV)}, y[ix].to(DEV)
    def ev(ix):
        net.eval(); ps=[]
        with torch.no_grad():
            for k in range(0,len(ix),512):
                b,_=bat(ix[k:k+512]); ps.append(net(b).argmax(-1).cpu().numpy())
        return f1_score(D["y"][ix], np.concatenate(ps), average="macro", zero_division=0)
    best,bstate,bad=-1,None,0
    for ep in range(epochs):
        net.train(); rr=np.random.RandomState(ep); o=tr.copy(); rr.shuffle(o)
        for k in range(0,len(o),bs):
            b,yb=bat(o[k:k+bs]); opt.zero_grad(); lf(net(b),yb).backward(); opt.step()
        sch.step(); vf=ev(va)
        if vf>best: best,bstate,bad=vf,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(bstate); return best, ev(te)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]); a=ap.parse_args()
    z=np.load(RAW, allow_pickle=True)
    pids=sorted(set(z["pids"].astype(str).tolist()))
    rng=np.random.RandomState(2026); rng.shuffle(pids)
    folds=[set(pids[i::a.folds]) for i in range(a.folds)]
    SUBS=[tuple(s) for r in (1,2,3) for s in itertools.combinations(MODS,r)]
    print(f"[CV MIMIC-Eye] N={len(z['y'])} 患者={len(pids)} folds={a.folds} seeds={a.seeds}",flush=True)
    acc={s:[] for s in SUBS}; sel=[]; bs1=[]; picks=[]
    for i in range(a.folds):
        D,tr,va,te = fold_data(z, folds, i)
        for sd in a.seeds:
            val={},; val={}; tst={}
            for s in SUBS:
                v,t = run(D,tr,va,te,s,sd); val[s]=v; tst[s]=t; acc[s].append(t)
            p=max(SUBS,key=lambda s: val[s]); sel.append(tst[p]); picks.append("+".join(p))
            bs1.append(max(tst[s] for s in SUBS if len(s)==1))
            print(f"  fold{i} seed{sd}: pick={'+'.join(p):18s} sel={tst[p]:.3f} | all3={tst[tuple(MODS)]:.3f} | best1={bs1[-1]:.3f} | time+text={tst[('time','text')]:.3f}",flush=True)
    def ms(x): return f"{st.mean(x):.3f}±{(st.pstdev(x) if len(x)>1 else 0):.3f}"
    print(f"\n=== 各子集 test macroF1(全部 {a.folds}x{len(a.seeds)} 次,无丢弃) ===",flush=True)
    for s in SUBS: print(f"  {'+'.join(s):20s} {ms(acc[s])}",flush=True)
    print(f"\n  >> SubsetSelect {ms(sel)} | best-single {ms(bs1)} | all-3 {ms(acc[tuple(MODS)])}",flush=True)
    print(f"  >> 单调性: Subset−best1 = {st.mean(sel)-st.mean(bs1):+.3f}",flush=True)
    from collections import Counter; print(f"  >> 选中: {dict(Counter(picks))}",flush=True)
    print("CV_DONE",flush=True)
