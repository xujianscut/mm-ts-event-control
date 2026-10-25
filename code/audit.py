"""
综合诊断审计(非循环证据):
  * 诚实 best-single(验证集选单模态) + FullConcat,配对 Δ_full,10seed,t区间,配对置换检验,example bootstrap
  * Ensemble-of-unimodal(独立单模态softmax平均) -> Δ_ens = Full - Ensemble(区分互补 vs 集成)
  * 条件置换检验:测试时打乱图像 -> FullConcat macro-F1 掉多少(图像条件信息的直接证据)
  * 误差重叠 / oracle(任一单模态对)
  * 自动检测在场模态(text 全零=不在场 -> 双模态 3 子集)
输出 AUDIT_JSON 一行,可直接进表。
"""
import sys, os, argparse, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
try:
    from train_v2_raiden import prepare, TimeEnc, ImgEnc, TxtEnc, DEV
except Exception:
    from train_v2 import prepare, TimeEnc, ImgEnc, TxtEnc, DEV

class Net(nn.Module):
    def __init__(s, mods, C, dtxt, nC):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); d=0
        for m in s.mods:
            e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dtxt)}[m](); s.enc[m]=e; d+=e.out
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.head(torch.cat([s.enc[m](b[m]) for m in s.mods],-1))

def present_mods(D):
    ms=["time","image"]; x=D["Xtxt"]
    if x.shape[1]>2 and float(x.std())>1e-6: ms.append("text")
    return ms

def tcrit(n): return {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}.get(n,1.96)

def run(D, mods, seed, epochs=60, bs=128, patience=12, want_proba=False, perm_img=False):
    torch.manual_seed(seed); np.random.seed(seed)
    sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); times=torch.tensor(D["times"]); imgs=torch.tensor(D["imgs"]).float()/255.; txt=torch.tensor(D["Xtxt"])
    C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Net(mods,C,D["Xtxt"].shape[1],nC).to(DEV)
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix, pimg=False):
        im=imgs[ix]
        if pimg:
            im=im[torch.randperm(len(ix))]
        return {"time":times[ix].to(DEV),"image":im.to(DEV),"text":txt[ix].to(DEV)},y[ix].to(DEV)
    def proba(ix, pimg=False):
        net.eval(); ps=[]
        with torch.no_grad():
            for k in range(0,len(ix),512):
                b,_=bat(ix[k:k+512],pimg); ps.append(torch.softmax(net(b),-1).cpu().numpy())
        return np.concatenate(ps)
    def ev(ix): return f1_score(D["y"][ix], proba(ix).argmax(1), average="macro", zero_division=0)
    best,state,bad=-1,None,0
    for ep in range(epochs):
        net.train(); rr=np.random.RandomState(ep); o=tr.copy(); rr.shuffle(o)
        for k in range(0,len(o),bs):
            b,yb=bat(o[k:k+bs]); opt.zero_grad(); lf(net(b),yb).backward(); opt.step()
        sch.step(); vf=ev(va)
        if vf>best: best,state,bad=vf,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(state)
    vf=ev(va); yte=D["y"][te]
    out={"val":vf,"test":f1_score(yte,proba(te).argmax(1),average="macro",zero_division=0)}
    if want_proba: out["proba"]=proba(te); out["y"]=yte
    if perm_img: out["test_permimg"]=f1_score(yte,proba(te,pimg=True).argmax(1),average="macro",zero_division=0)
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=present_mods(D); nC=int(D["nC"])
    print(f"[audit {a.dataset}] N={len(D['y'])} present={pm} lattice={2**len(pm)-1} nC={nC} seeds={len(a.seeds)}",flush=True)
    single_val={m:[] for m in pm}; single_te={m:[] for m in pm}; full_te=[]; full_perm=[]
    seed0={}
    for si,sd in enumerate(a.seeds):
        for m in pm:
            r=run(D,[m],sd,want_proba=(si==0))
            single_val[m].append(r["val"]); single_te[m].append(r["test"])
            if si==0: seed0[m]=r["proba"]; seed0["y"]=r["y"]
        rf=run(D,pm,sd,want_proba=(si==0),perm_img=True)
        full_te.append(rf["test"]); full_perm.append(rf["test_permimg"])
        if si==0: seed0["full"]=rf["proba"]
    n=len(a.seeds)
    bestmod=max(pm,key=lambda m: np.mean(single_val[m]))
    bs_te=np.array(single_te[bestmod]); fu=np.array(full_te); fp=np.array(full_perm)
    # Δ_full (paired)
    d=fu-bs_te; dm=d.mean(); sd_=d.std(ddof=1) if n>1 else 0; se=sd_/np.sqrt(n); ci=tcrit(n)*se
    rng=np.random.RandomState(0); B=20000
    p_perm=(sum(abs((d*rng.choice([-1,1],size=n)).mean())>=abs(dm) for _ in range(B))+1)/(B+1)
    # Ensemble of unimodal (seed0 softmax avg)
    y0=seed0["y"]; ens=np.mean([seed0[m] for m in pm],axis=0).argmax(1)
    ens_f1=f1_score(y0,ens,average="macro",zero_division=0)
    full_f1_s0=f1_score(y0,seed0["full"].argmax(1),average="macro",zero_division=0)
    # oracle (any unimodal correct) + error overlap
    correct={m:(seed0[m].argmax(1)==y0) for m in pm}
    oracle=np.any([correct[m] for m in pm],axis=0).mean()
    # all-wrong-together rate (低=互补)
    all_wrong=np.mean(~np.any([correct[m] for m in pm],axis=0))
    # 条件置换:图像打乱后的掉幅
    perm_drop=(fu-fp).mean()
    # example bootstrap of Δ_full (seed0)
    bf=seed0["full"].argmax(1); bb=seed0[bestmod].argmax(1); idx=np.arange(len(y0)); bt=[]
    for _ in range(5000):
        r=rng.choice(idx,size=len(idx),replace=True)
        bt.append(f1_score(y0[r],bf[r],average="macro",zero_division=0)-f1_score(y0[r],bb[r],average="macro",zero_division=0))
    bt=np.sort(bt); blo,bhi=bt[125],bt[4874]
    res={"ds":a.dataset,"present":pm,"best":bestmod,
         "best_f1":round(float(bs_te.mean()),3),"best_sd":round(float(bs_te.std(ddof=1) if n>1 else 0),3),
         "full_f1":round(float(fu.mean()),3),"full_sd":round(float(fu.std(ddof=1) if n>1 else 0),3),
         "delta":round(float(dm),3),"ci":round(float(ci),3),"perm_p":round(float(p_perm),3),
         "boot95":[round(float(blo),3),round(float(bhi),3)],
         "ensemble_f1":round(float(ens_f1),3),"delta_vs_ens":round(float(full_f1_s0-ens_f1),3),
         "oracle":round(float(oracle),3),"all_wrong":round(float(all_wrong),3),
         "img_perm_drop":round(float(perm_drop),3),
         "pos_rate":round(float(np.mean(D["y"])),3)}
    print("  best-single(%s) %.3f±%.3f | FullConcat %.3f±%.3f"%(bestmod,bs_te.mean(),res["best_sd"],fu.mean(),res["full_sd"]),flush=True)
    print("  Δ_full=%+.3f  t%d-95%%CI[%+.3f,%+.3f]  perm-p=%.3f  boot95[%+.3f,%+.3f]"%(dm,n-1,dm-ci,dm+ci,p_perm,blo,bhi),flush=True)
    print("  Ensemble=%.3f  Δ(Full-Ens)=%+.3f  | oracle=%.3f all-wrong=%.3f | image-perm drop=%+.3f"%(ens_f1,res["delta_vs_ens"],oracle,all_wrong,perm_drop),flush=True)
    print("AUDIT_JSON "+json.dumps(res),flush=True)
