"""统一一致审计:一个RNG流里 per-seed 训练各单模态+FullConcat,算 ensemble;Δ_full 和 Δ_ens 对同一 best-single;t区间+置换检验;+图像置换掉幅+oracle。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
try: from train_v2_raiden import prepare,TimeEnc,ImgEnc,TxtEnc,DEV
except Exception: from train_v2 import prepare,TimeEnc,ImgEnc,TxtEnc,DEV
class Net(nn.Module):
    def __init__(s,mods,C,dt,nC):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); d=0
        for m in s.mods:
            e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.enc[m]=e; d+=e.out
        s.h=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(torch.cat([s.enc[m](b[m]) for m in s.mods],-1))
def present(D):
    ms=["time","image"]; x=D["Xtxt"]
    if x.shape[1]>2 and float(x.std())>1e-6: ms.append("text")
    return ms
def tcrit(n): return {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}.get(n,1.96)
def train(D,mods,seed,ep=60,bs=128,pat=12,perm=False):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; tx=torch.tensor(D["Xtxt"]); C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Net(mods,C,D["Xtxt"].shape[1],nC).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix,pi=False):
        I=im[ix]
        if pi: I=I[torch.randperm(len(ix))]
        return {"time":ti[ix].to(DEV),"image":I.to(DEV),"text":tx[ix].to(DEV)},y[ix].to(DEV)
    def prob(ix,pi=False):
        net.eval(); P=[]
        with torch.no_grad():
            for k in range(0,len(ix),512): b,_=bat(ix[k:k+512],pi); P.append(torch.softmax(net(b),-1).cpu().numpy())
        return np.concatenate(P)
    def ev(ix): return f1_score(D["y"][ix],prob(ix).argmax(1),average="macro",zero_division=0)
    best,stt,bad=-1,None,0
    for e in range(ep):
        net.train(); rr=np.random.RandomState(1000+e); o=tr.copy(); rr.shuffle(o)   # 固定独立RNG,不受下游影响
        for k in range(0,len(o),bs): b,yb=bat(o[k:k+bs]); opt.zero_grad(); lf(net(b),yb).backward(); opt.step()
        sch.step(); vf=ev(va)
        if vf>best: best,stt,bad=vf,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=pat: break
    net.load_state_dict(stt)
    out={"val":ev(va),"proba":prob(te),"y":D["y"][te]}
    if perm: out["permf1"]=f1_score(out["y"],prob(te,pi=True).argmax(1),average="macro",zero_division=0)
    return out
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=present(D); nC=int(D["nC"]); n=len(a.seeds)
    print(f"[audit2 {a.dataset}] present={pm} nC={nC} test={int((D['sp']=='test').sum())}",flush=True)
    sv={m:[] for m in pm}; st={m:[] for m in pm}; full=[]; ens=[]; permdrop=[]; s0={}
    for si,sd in enumerate(a.seeds):
        pr={}
        for m in pm:
            r=train(D,[m],sd); pr[m]=r["proba"]; yt=r["y"]; sv[m].append(r["val"]); st[m].append(f1_score(yt,r["proba"].argmax(1),average="macro",zero_division=0))
            if si==0: s0[m]=r["proba"]; s0["y"]=yt
        rf=train(D,pm,sd,perm=True); ff=f1_score(rf["y"],rf["proba"].argmax(1),average="macro",zero_division=0)
        full.append(ff); permdrop.append(ff-rf["permf1"])
        ens.append(f1_score(rf["y"],np.mean([pr[m] for m in pm],0).argmax(1),average="macro",zero_division=0))
        if si==0: s0["full"]=rf["proba"]
    bm=max(pm,key=lambda m:np.mean(sv[m])); bs=np.array(st[bm]); fu=np.array(full); en=np.array(ens)
    def stat(x):
        d=x-bs; dm=d.mean(); s=d.std(ddof=1); se=s/np.sqrt(n); ci=tcrit(n)*se
        rng=np.random.RandomState(0); B=20000
        p=(sum(abs((d*rng.choice([-1,1],size=n)).mean())>=abs(dm) for _ in range(B))+1)/(B+1)
        return round(float(dm),3),round(float(ci),3),round(float(p),3)
    df=stat(fu); de=stat(en)
    y0=s0["y"]; corr={m:(s0[m].argmax(1)==y0) for m in pm}; oracle=float(np.any([corr[m] for m in pm],0).mean())
    res={"ds":a.dataset,"present":pm,"testN":int(len(y0)),"best":bm,
         "best_f1":round(float(bs.mean()),3),"best_sd":round(float(bs.std(ddof=1)),3),
         "full_f1":round(float(fu.mean()),3),"delta_full":df[0],"ci_full":df[1],"p_full":df[2],
         "ens_f1":round(float(en.mean()),3),"delta_ens":de[0],"ci_ens":de[1],"p_ens":de[2],
         "img_perm_drop":round(float(np.mean(permdrop)),3),"oracle":round(oracle,3),"pos_rate":round(float(np.mean(D["y"])),3)}
    print("  best(%s)=%.3f | Full=%.3f Δ=%+.3f(±%.3f,p=%.3f) | Ens=%.3f Δ=%+.3f(±%.3f,p=%.3f) | imgdrop=%+.3f"%(
        bm,bs.mean(),fu.mean(),df[0],df[1],df[2],en.mean(),de[0],de[1],de[2],np.mean(permdrop)),flush=True)
    print("AUDIT2_JSON "+json.dumps(res),flush=True)
