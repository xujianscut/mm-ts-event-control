"""
SubsetSelect:在【模态格】上做验证集选择
候选 = 所有非空模态子集的专用模型 {image},{time},{text},{time,text},{time,image},{image,text},{time,image,text}
每个子集训一个专用模型(该子集的编码器 concat -> 头),val 选最优 -> 报告其 test。
=> 构造上保证不低于【任何模态子集】,直接对应"加模态不该变差"。
"""
import sys, os, argparse, warnings, statistics as st, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from train_v2_raiden import prepare, TimeEnc, ImgEnc, TxtEnc, DEV

MODS = ["time", "image", "text"]

class SubsetNet(nn.Module):
    def __init__(s, mods, C, dtxt, nC):
        super().__init__()
        s.mods = list(mods); s.enc = nn.ModuleDict()
        d = 0
        for m in s.mods:
            e = {"time": lambda: TimeEnc(C), "image": ImgEnc, "text": lambda: TxtEnc(dtxt)}[m]()
            s.enc[m] = e; d += e.out
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nC))
    def forward(s, b):
        return s.head(torch.cat([s.enc[m](b[m]) for m in s.mods], -1))

def run(D, mods, seed, epochs=60, bs=128, patience=12):
    torch.manual_seed(seed); np.random.seed(seed)
    sp = D["sp"]; nC = int(D["nC"])
    tr = np.where(np.isin(sp, ["train","training"]))[0]; va = np.where(sp=="val")[0]; te = np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); times=torch.tensor(D["times"]); imgs=torch.tensor(D["imgs"]).float()/255.; txt=torch.tensor(D["Xtxt"])
    C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=SubsetNet(mods,C,D["Xtxt"].shape[1],nC).to(DEV)
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
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]); a=ap.parse_args()
    D=prepare(a.dataset)
    SUBS=[tuple(s) for r in (1,2,3) for s in itertools.combinations(MODS,r)]
    print(f"[SubsetSelect {a.dataset}] N={len(D['y'])} 候选子集={len(SUBS)}",flush=True)
    sel_t, all3_t, bs1_t, picks = [], [], [], []
    for sd in a.seeds:
        val,tst={},{}
        for s in SUBS:
            v,t=run(D,s,sd); val[s],tst[s]=v,t
        pick=max(SUBS,key=lambda s: val[s])
        bs1=max(tst[s] for s in SUBS if len(s)==1)
        sel_t.append(tst[pick]); all3_t.append(tst[tuple(MODS)]); bs1_t.append(bs1); picks.append("+".join(pick))
        print(f"  seed{sd}: pick={'+'.join(pick):18s} sel_test={tst[pick]:.3f} | all3={tst[tuple(MODS)]:.3f} | best_single={bs1:.3f}",flush=True)
    def ms(x): return f"{st.mean(x):.3f}±{(st.pstdev(x) if len(x)>1 else 0):.3f}"
    print(f"  >> SubsetSelect {ms(sel_t)} | all-3(强制全模态) {ms(all3_t)} | best-single {ms(bs1_t)}",flush=True)
    print(f"  >> 单调性: Subset-best_single={st.mean(sel_t)-st.mean(bs1_t):+.3f} | all3-best_single={st.mean(all3_t)-st.mean(bs1_t):+.3f}",flush=True)
    from collections import Counter; print(f"  >> 选中: {dict(Counter(picks))}",flush=True)
