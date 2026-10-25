"""强编码器稳健性:深残差CNN + 2层BiGRU(h=256)替换弱编码器,
   重算 best/FullConcat/Ensemble/size-matched control + 聚类/例bootstrap。
   目的:验证 ensemble 的净互补(Δens-Δctrl)不依赖弱编码器。核心指标,略去shaspec/timecap/wfusion。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from train_v2_raiden import prepare, DEV

class StrongTimeEnc(nn.Module):
    def __init__(s,C,h=256):
        super().__init__(); s.gru=nn.GRU(C,h,2,batch_first=True,bidirectional=True,dropout=0.1); s.out=2*h
    def forward(s,x): o=s.gru(x)[0]; return o.mean(1)
class ResBlock(nn.Module):
    def __init__(s,ci,co,stride=1):
        super().__init__()
        s.c1=nn.Conv2d(ci,co,3,stride,1,bias=False); s.b1=nn.BatchNorm2d(co)
        s.c2=nn.Conv2d(co,co,3,1,1,bias=False); s.b2=nn.BatchNorm2d(co)
        s.sc=nn.Sequential() if (stride==1 and ci==co) else nn.Sequential(nn.Conv2d(ci,co,1,stride,bias=False),nn.BatchNorm2d(co))
    def forward(s,x): h=F.relu(s.b1(s.c1(x))); h=s.b2(s.c2(h)); return F.relu(h+s.sc(x))
class StrongImgEnc(nn.Module):
    def __init__(s):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv2d(3,32,3,1,1,bias=False),nn.BatchNorm2d(32),nn.ReLU())
        s.l1=ResBlock(32,64,2); s.l2=ResBlock(64,128,2); s.l3=ResBlock(128,256,2); s.l4=ResBlock(256,256,2)
        s.out=256
    def forward(s,x): h=s.l4(s.l3(s.l2(s.l1(s.stem(x))))); return F.adaptive_avg_pool2d(h,1).flatten(1)

class Net(nn.Module):
    def __init__(s,mods,C,nC):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); d=0
        for m in s.mods:
            e=StrongTimeEnc(C) if m=="time" else StrongImgEnc(); s.enc[m]=e; d+=e.out
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.head(torch.cat([s.enc[m](b[m]) for m in s.mods],-1))

def tcrit(n): return {2:12.706,5:2.776,8:2.365,10:2.262}.get(n,1.96)
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)

def train(D,mods,seed,epochs=50,bs=128,patience=10):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Net(mods,C,nC).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix): return {"time":ti[ix].to(DEV),"image":im[ix].to(DEV)},y[ix].to(DEV)
    def prob(ix):
        net.eval(); P=[]
        with torch.no_grad():
            for k in range(0,len(ix),256): b,_=bat(ix[k:k+256]); P.append(torch.softmax(net(b),-1).cpu().numpy())
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
    net.load_state_dict(stt)
    return {"val_f1":macro(D["y"][va],prob(va).argmax(1)),"te_prob":prob(te)}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=["time","image"]; nC=int(D["nC"]); n=len(a.seeds)
    sp=D["sp"]; te=np.where(sp=="test")[0]; yte=D["y"][te]; k=len(pm)
    gidf=f"/home/jianxu/gid_{a.dataset}.npy"; gid=np.load(gidf,allow_pickle=True).astype(str)[te] if os.path.exists(gidf) else None
    print(f"[strong {a.dataset}] STRONG enc testN={len(te)} grouped={gid is not None} seeds={n}",flush=True)
    sv={m:[] for m in pm}; ste={m:[] for m in pm}; fte=[]; ete=[]; cte=[]
    sum_best=None; sum_ens=None
    for sd in a.seeds:
        tp={}
        for m in pm:
            r=train(D,[m],sd); sv[m].append(r["val_f1"]); tp[m]=r["te_prob"]; ste[m].append(macro(yte,tp[m].argmax(1)))
        bestmod=max(pm,key=lambda m: sv[m][-1])
        rf=train(D,pm,sd); fte.append(macro(yte,rf["te_prob"].argmax(1)))
        ens_p=np.mean([tp[m] for m in pm],0); ete.append(macro(yte,ens_p.argmax(1)))
        cps=[tp[bestmod]]+[train(D,[bestmod],sd+1000*(j+1))["te_prob"] for j in range(k-1)]
        cte.append(macro(yte,np.mean(cps,0).argmax(1)))
        sum_best=tp[bestmod] if sum_best is None else sum_best+tp[bestmod]; sum_ens=ens_p if sum_ens is None else sum_ens+ens_p
        print(f"  seed{sd}: best={bestmod} single={ste[bestmod][-1]:.3f} full={fte[-1]:.3f} ens={ete[-1]:.3f} ctrl={cte[-1]:.3f}",flush=True)
    bestmod=max(pm,key=lambda m: np.mean(sv[m]))
    bs=np.array(ste[bestmod]); fu=np.array(fte); en=np.array(ete); ct=np.array(cte)
    def dstat(arr):
        d=arr-bs; dm=d.mean(); ci=tcrit(n)*d.std(ddof=1)/np.sqrt(n)
        rng=np.random.RandomState(0); B=20000
        p=(sum(abs((d*rng.choice([-1,1],size=n)).mean())>=abs(dm) for _ in range(B))+1)/(B+1)
        return round(float(dm),3),round(float(ci),3),round(float(p),3)
    mb=(sum_best/n).argmax(1); me=(sum_ens/n).argmax(1)
    rng=np.random.RandomState(1); diffs=[]
    if gid is not None:
        gs=np.array(sorted(set(gid))); gm={g:np.where(gid==g)[0] for g in gs}
        for _ in range(2000):
            ch=rng.choice(gs,len(gs),replace=True); idx=np.concatenate([gm[g] for g in ch]); diffs.append(macro(yte[idx],me[idx])-macro(yte[idx],mb[idx]))
    else:
        ia=np.arange(len(yte))
        for _ in range(2000):
            idx=rng.choice(ia,len(ia),replace=True); diffs.append(macro(yte[idx],me[idx])-macro(yte[idx],mb[idx]))
    diffs=np.sort(diffs); blo,bhi=float(diffs[49]),float(diffs[1949]); bp=float((diffs<=0).mean())
    res={"ds":a.dataset,"enc":"STRONG","best":bestmod,"best_f1":round(float(bs.mean()),3),
         "full_f1":round(float(fu.mean()),3),"dfull":dstat(fu),"ens_f1":round(float(en.mean()),3),"dens":dstat(en),
         "ctrl_f1":round(float(ct.mean()),3),"dctrl":dstat(ct),"net_compl":round(float(dstat(en)[0]-dstat(ct)[0]),3),
         "boot_dens_ci":[round(blo,3),round(bhi,3)],"boot_p":round(bp,3),"grouped":gid is not None}
    print("STRONG_JSON "+json.dumps(res),flush=True); print(f"STRONG_DONE_{a.dataset}",flush=True)
