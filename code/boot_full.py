"""给气旋 concat 增益(Δfull)也做 storm 聚类 bootstrap(此前只对Δens做)。
   训 time/image 单模态 + FullConcat,seed平均proba,对 Δfull 和 Δens 都做聚类bootstrap。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from train_v2_raiden import prepare,TimeEnc,ImgEnc,DEV
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)
class Single(nn.Module):
    def __init__(s,m,C,nC):
        super().__init__(); e=TimeEnc(C) if m=="time" else ImgEnc(); s.e=e; s.m=m
        s.h=nn.Sequential(nn.LayerNorm(e.out),nn.Linear(e.out,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(s.e(b[s.m]))
class Concat(nn.Module):
    def __init__(s,C,nC):
        super().__init__(); s.t=TimeEnc(C); s.i=ImgEnc(); d=s.t.out+s.i.out
        s.h=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(torch.cat([s.t(b["time"]),s.i(b["image"])],-1))
def train(D,kind,m,seed,epochs=60,bs=128,patience=12):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=(Single(m,C,nC) if kind=="single" else Concat(C,nC)).to(DEV)
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix): return {"time":ti[ix].to(DEV),"image":im[ix].to(DEV)},y[ix].to(DEV)
    def prob(ix):
        net.eval(); P=[]
        with torch.no_grad():
            for k in range(0,len(ix),512): b,_=bat(ix[k:k+512]); P.append(torch.softmax(net(b),-1).cpu().numpy())
        return np.concatenate(P)
    best,stt,bad=-1,None,0
    for e in range(epochs):
        net.train(); rr=np.random.RandomState(2000+e); o=tr.copy(); rr.shuffle(o)
        for k in range(0,len(o),bs): b,yb=bat(o[k:k+bs]); opt.zero_grad(); lf(net(b),yb).backward(); opt.step()
        sch.step(); vf=macro(D["y"][va],prob(va).argmax(1))
        if vf>best: best,stt,bad=vf,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(stt); return macro(D["y"][va],prob(va).argmax(1)), prob(te)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); te=np.where(D["sp"]=="test")[0]; yte=D["y"][te]; n=len(a.seeds)
    gid=np.load(f"/home/jianxu/gid_{a.dataset}.npy",allow_pickle=True).astype(str)[te]
    print(f"[boot_full {a.dataset}] testN={len(te)} groups={len(set(gid))} seeds={n}",flush=True)
    sv={"time":[],"image":[]}; sump={"time":None,"image":None}; sumf=None
    for sd in a.seeds:
        for m in ("time","image"):
            v,p=train(D,"single",m,sd); sv[m].append(v); sump[m]=p if sump[m] is None else sump[m]+p
        _,pf=train(D,"concat",None,sd); sumf=pf if sumf is None else sumf+pf
        print(f"  seed{sd} done",flush=True)
    bestmod=max(("time","image"),key=lambda m:np.mean(sv[m]))
    mb=(sump[bestmod]/n).argmax(1); mf=(sumf/n).argmax(1); me=((sump["time"]+sump["image"])/(2*n)).argmax(1)
    gs=np.array(sorted(set(gid))); gm={g:np.where(gid==g)[0] for g in gs}; rng=np.random.RandomState(1)
    df=[]; de=[]
    for _ in range(2000):
        ch=rng.choice(gs,len(gs),replace=True); idx=np.concatenate([gm[g] for g in ch])
        bb=macro(yte[idx],mb[idx]); df.append(macro(yte[idx],mf[idx])-bb); de.append(macro(yte[idx],me[idx])-bb)
    df=np.sort(df); de=np.sort(de)
    res={"ds":a.dataset,"best":bestmod,"boot_dfull_ci":[round(float(df[49]),3),round(float(df[1949]),3)],"boot_dfull_p":round(float((df<=0).mean()),3),
         "boot_dens_ci":[round(float(de[49]),3),round(float(de[1949]),3)],"boot_dens_p":round(float((de<=0).mean()),3),"nGroups":len(gs)}
    print("BOOTFULL_JSON "+json.dumps(res),flush=True); print(f"BOOTFULL_DONE_{a.dataset}",flush=True)
