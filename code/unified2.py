"""统一 10-seed 实验(单一 RNG 流,所有数值共享同一 best-single):
  singles(+val/test proba) / FullConcat(+aux条件置换) / ShaSpec* / TimeCAP* /
  validation-learned 加权 late-fusion(可abstain) / uniform ensemble /
  size-matched 同模态 control(k=模态数个best副本) / oracle / error-overlap
  显著性:seed 符号置换(=跨训练种子稳定性) + storm/例级 bootstrap(=测试采样)
输出 UNI_JSON 一行。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from train_v2_raiden import prepare,TimeEnc,ImgEnc,TxtEnc,DEV
try:
    from shaspec_raiden import run as shaspec_run
except Exception: shaspec_run=None
try:
    from tstext_raiden import run as tstext_run
except Exception: tstext_run=None

class Net(nn.Module):
    def __init__(s,mods,C,dtxt,nC):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); d=0
        for m in s.mods:
            e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dtxt)}[m](); s.enc[m]=e; d+=e.out
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(d if False else 128,nC))
    def forward(s,b): return s.head(torch.cat([s.enc[m](b[m]) for m in s.mods],-1))

def present_mods(D):
    ms=["time","image"]; x=D["Xtxt"]
    if x.shape[1]>2 and float(x.std())>1e-6: ms.append("text")
    return ms
def tcrit(n): return {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}.get(n,1.96)
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)

def train(D,mods,seed,epochs=60,bs=128,patience=12,perm_aux=False):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; tx=torch.tensor(D["Xtxt"]); C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Net(mods,C,D["Xtxt"].shape[1],nC).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix,pa=False):
        b={"time":ti[ix].to(DEV),"image":im[ix].to(DEV),"text":tx[ix].to(DEV)}
        if pa: b[pa]=b[pa][torch.randperm(len(ix))]
        return b,y[ix].to(DEV)
    def prob(ix,pa=False):
        net.eval(); P=[]
        with torch.no_grad():
            for k in range(0,len(ix),512): b,_=bat(ix[k:k+512],pa); P.append(torch.softmax(net(b),-1).cpu().numpy())
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
    out={"val_f1":macro(D["y"][va],prob(va).argmax(1)),"te_f1":macro(D["y"][te],prob(te).argmax(1)),
         "va_prob":prob(va),"te_prob":prob(te)}
    if perm_aux:
        out["aux_drop"]={m:out["te_f1"]-macro(D["y"][te],prob(te,pa=m).argmax(1)) for m in mods if m!="time" or len(mods)>1}
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=present_mods(D); nC=int(D["nC"]); n=len(a.seeds)
    sp=D["sp"]; te=np.where(sp=="test")[0]; yte=D["y"][te]; k=len(pm)
    gidf=f"/home/jianxu/gid_{a.dataset}.npy"; gid=np.load(gidf,allow_pickle=True).astype(str)[te] if os.path.exists(gidf) else None
    print(f"[unified2 {a.dataset}] present={pm} testN={len(te)} grouped={gid is not None} seeds={n}",flush=True)
    single_te={m:[] for m in pm}; single_va={m:[] for m in pm}; tp_all={m:[] for m in pm}
    full_te=[]; aux_acc={m:[] for m in pm}; sha_te=[]; tc_te=[]; wf_te=[]; ens_te=[]
    for sd in a.seeds:
        vp={}; tp={}
        for m in pm:
            r=train(D,[m],sd); single_te[m].append(r["te_f1"]); single_va[m].append(r["val_f1"]); vp[m]=r["va_prob"]; tp[m]=r["te_prob"]; tp_all[m].append(r["te_prob"])
        rf=train(D,pm,sd,perm_aux=True); full_te.append(rf["te_f1"])
        for m,dp in rf["aux_drop"].items(): aux_acc[m].append(dp)
        if shaspec_run is not None:
            try: sha_te.append(float(np.atleast_1d(shaspec_run(D,sd))[0]))
            except Exception: pass
        if tstext_run is not None and "text" in pm:
            try: tc_te.append(float(np.atleast_1d(tstext_run(D,sd))[0]))
            except Exception: pass
        ens_p=np.mean([tp[m] for m in pm],0); ens_te.append(macro(yte,ens_p.argmax(1)))
        yva=D["y"][np.where(sp=="val")[0]] if (sp=="val").any() else None
        if yva is not None and len(yva)==len(vp[pm[0]]):
            w=np.ones(k)/k
            def f1w(w,probs,y): return macro(y,sum(w[i]*probs[pm[i]] for i in range(k)).argmax(1))
            bestv=f1w(w,vp,yva)
            for _ in range(300):
                imp=False
                for i in range(k):
                    for dl in (0.15,-0.15,0.3,-0.3):
                        w2=w.copy(); w2[i]=max(0,w2[i]+dl); s2=w2.sum()
                        if s2<=0: continue
                        w2/=s2; v=f1w(w2,vp,yva)
                        if v>bestv+1e-6: w,bestv=w2,v; imp=True
                if not imp: break
            wf_te.append(macro(yte,sum(w[i]*tp[pm[i]] for i in range(k)).argmax(1)))
        print(f"  seed{sd} done",flush=True)
    # 全局 best = test-max 单模态(最保守,消除val/test歧义)
    bestmod=max(pm,key=lambda m: np.mean(single_te[m]))
    # control:k个best模态副本(不同init),per-seed
    ctrl_te=[]
    for si,sd in enumerate(a.seeds):
        cps=[tp_all[bestmod][si]]+[train(D,[bestmod],sd+1000*(j+1))["te_prob"] for j in range(k-1)]
        ctrl_te.append(macro(yte,np.mean(cps,0).argmax(1)))
    bs=np.array(single_te[bestmod]); fu=np.array(full_te); en=np.array(ens_te); ct=np.array(ctrl_te)
    def tcrit(nn): return {2:12.706,5:2.776,8:2.365,10:2.262}.get(nn,1.96)
    def perm_p(d):
        dm=d.mean(); rng=np.random.RandomState(0)
        return (sum(abs((d*rng.choice([-1,1],size=len(d))).mean())>=abs(dm) for _ in range(20000))+1)/20001
    def dstat(arr):
        d=arr-bs; dm=d.mean(); ci=tcrit(n)*d.std(ddof=1)/np.sqrt(n)
        return round(float(dm),3),round(float(ci),3),round(float(perm_p(d)),3)
    dfull=dstat(fu); dens=dstat(en); dctrl=dstat(ct); dwf=dstat(np.array(wf_te)) if len(wf_te)==n else None
    # ★per-seed excess = ens - ctrl 的配对检验
    exc=en-ct; exc_p=round(float(perm_p(exc)),3); exc_m=round(float(exc.mean()),3); exc_ci=round(float(tcrit(n)*exc.std(ddof=1)/np.sqrt(n)),3)
    # bootstrap on Δens (cluster if gid)
    sum_best=np.mean(tp_all[bestmod],0); sum_ens=np.mean([np.mean(tp_all[m],0) for m in pm],0)
    mb=sum_best.argmax(1); me=sum_ens.argmax(1); rng=np.random.RandomState(1); diffs=[]
    if gid is not None:
        gs=np.array(sorted(set(gid))); gm={g:np.where(gid==g)[0] for g in gs}
        for _ in range(2000):
            ch=rng.choice(gs,len(gs),replace=True); idx=np.concatenate([gm[g] for g in ch]); diffs.append(macro(yte[idx],me[idx])-macro(yte[idx],mb[idx]))
    else:
        ia=np.arange(len(yte))
        for _ in range(2000):
            idx=rng.choice(ia,len(ia),replace=True); diffs.append(macro(yte[idx],me[idx])-macro(yte[idx],mb[idx]))
    diffs=np.sort(diffs); blo,bhi=float(diffs[49]),float(diffs[1949]); bp=float((diffs<=0).mean())
    corr={m:(np.mean(tp_all[m],0).argmax(1)==yte) for m in pm}
    oracle=float(np.any([corr[m] for m in pm],0).mean()); allwrong=float(np.mean(~np.any([corr[m] for m in pm],0)))
    res={"ds":a.dataset,"present":pm,"testN":int(len(te)),"grouped":gid is not None,"best":bestmod,
         "best_f1":round(float(bs.mean()),3),"singles":{m:round(float(np.mean(single_te[m])),3) for m in pm},
         "val_best":max(pm,key=lambda m: np.mean(single_va[m])),
         "full_f1":round(float(fu.mean()),3),"dfull":dfull,"ens_f1":round(float(en.mean()),3),"dens":dens,
         "ctrl_f1":round(float(ct.mean()),3),"dctrl":dctrl,"excess":exc_m,"excess_ci":exc_ci,"excess_p":exc_p,
         "wf_f1":(round(float(np.mean(wf_te)),3) if wf_te else None),"dwf":dwf,
         "shaspec_f1":(round(float(np.mean(sha_te)),3) if sha_te else None),
         "timecap_f1":(round(float(np.mean(tc_te)),3) if tc_te else None),
         "aux_drop":{m:round(float(np.mean(v)),3) for m,v in aux_acc.items() if v},
         "oracle":round(oracle,3),"all_wrong":round(allwrong,3),
         "boot_dens_ci":[round(blo,3),round(bhi,3)],"boot_p":round(bp,4)}
    print("UNI2_JSON "+json.dumps(res),flush=True); print(f"UNI2_DONE_{a.dataset}",flush=True)
