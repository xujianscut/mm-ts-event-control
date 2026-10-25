"""MIMIC-Eye 5-fold grouped-CV(按患者):best-single/FullConcat/Ensemble/same-mod control,
   给主表提供 CV 版数字(回应 reviewer:主表勿用 single-split 正结果)。"""
import sys,os,warnings,statistics as st,json
warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.expanduser("~"))
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from train_v2_raiden import TimeEnc,ImgEnc,TxtEnc,DEV
from cv_me import fold_data
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)
class Net(nn.Module):
    def __init__(s,mods,C,dt,nC):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); d=0
        for m in s.mods:
            e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.enc[m]=e; d+=e.out
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.head(torch.cat([s.enc[m](b[m]) for m in s.mods],-1))
def run(D,tr,va,te,mods,seed,epochs=60,bs=128,patience=12):
    torch.manual_seed(seed); np.random.seed(seed); nC=int(D["nC"])
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; tx=torch.tensor(D["Xtxt"]); C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Net(mods,C,D["Xtxt"].shape[1],nC).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix): return {"time":ti[ix].to(DEV),"image":im[ix].to(DEV),"text":tx[ix].to(DEV)},y[ix].to(DEV)
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
    RAW=os.path.expanduser("~/v2_cache/mimiceye_raw.npz"); z=np.load(RAW,allow_pickle=True)
    pids=sorted(set(z["pids"].astype(str).tolist())); rng=np.random.RandomState(2026); rng.shuffle(pids)
    F=5; folds=[set(pids[i::F]) for i in range(F)]; seeds=[2026,2027,2028,2029,2030]; pm=["time","image","text"]
    print(f"[MIMIC-CV] N={len(z['y'])} patients={len(pids)} folds={F}",flush=True)
    bestL=[]; fullL=[]; ensL=[]; ctrlL=[]
    for i in range(F):
        D,tr,va,te=fold_data(z,folds,i); yte=D["y"][te]
        for sd in seeds:
            tp={}
            for m in pm: _,tp[m]=run(D,tr,va,te,[m],sd)
            singles={m:macro(yte,tp[m].argmax(1)) for m in pm}
            bm=max(pm,key=lambda m:singles[m]); best=singles[bm]; bestL.append(best)
            _,pf=run(D,tr,va,te,pm,sd); fullL.append(macro(yte,pf.argmax(1))-best)
            ens=np.mean([tp[m] for m in pm],0); ensL.append(macro(yte,ens.argmax(1))-best)
            cps=[tp[bm]]+[run(D,tr,va,te,[bm],sd+500*(j+1))[1] for j in range(len(pm)-1)]
            ctrlL.append(macro(yte,np.mean(cps,0).argmax(1))-best)
            print(f"  fold{i} seed{sd}: best={best:.3f} dfull={fullL[-1]:+.3f} dens={ensL[-1]:+.3f} dctrl={ctrlL[-1]:+.3f}",flush=True)
    res={"best_f1":round(st.mean(bestL),3),"dfull":round(st.mean(fullL),3),"dens":round(st.mean(ensL),3),
         "dctrl":round(st.mean(ctrlL),3),"excess":round(st.mean(ensL)-st.mean(ctrlL),3),"nFoldSeed":len(bestL)}
    print("MIMICCV_JSON "+json.dumps(res),flush=True); print("MIMICCV_DONE",flush=True)
