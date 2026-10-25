"""同模态集成对照:2个不同初始化的最强单模态模型集成(隔离方差缩减)。
   若 跨模态ens增益 >> 同模态ens增益,则增益=真互补,非bagging。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
try: from train_v2_raiden import prepare,TimeEnc,ImgEnc,TxtEnc,DEV
except Exception: from train_v2 import prepare,TimeEnc,ImgEnc,TxtEnc,DEV
class Net(nn.Module):
    def __init__(s,m,C,dt,nC):
        super().__init__(); e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.e=e; s.m=m
        s.h=nn.Sequential(nn.LayerNorm(e.out),nn.Linear(e.out,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(s.e(b[s.m]))
def present(D):
    ms=["time","image"]; x=D["Xtxt"]
    if x.shape[1]>2 and float(x.std())>1e-6: ms.append("text")
    return ms
def tcrit(n): return {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}.get(n,1.96)
def train(D,m,seed,ep=60,bs=128,pat=12):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; tx=torch.tensor(D["Xtxt"]); C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Net(m,C,D["Xtxt"].shape[1],nC).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix): return {"time":ti[ix].to(DEV),"image":im[ix].to(DEV),"text":tx[ix].to(DEV)},y[ix].to(DEV)
    def prob(ix):
        net.eval(); P=[]
        with torch.no_grad():
            for k in range(0,len(ix),512): b,_=bat(ix[k:k+512]); P.append(torch.softmax(net(b),-1).cpu().numpy())
        return np.concatenate(P)
    def ev(ix): return f1_score(D["y"][ix],prob(ix).argmax(1),average="macro",zero_division=0)
    best,stt,bad=-1,None,0
    for e in range(ep):
        net.train(); rr=np.random.RandomState(2000+e); o=tr.copy(); rr.shuffle(o)
        for k in range(0,len(o),bs): b,yb=bat(o[k:k+bs]); opt.zero_grad(); lf(net(b),yb).backward(); opt.step()
        sch.step(); vf=ev(va)
        if vf>best: best,stt,bad=vf,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=pat: break
    net.load_state_dict(stt); return ev(va),prob(te),D["y"][te]
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=present(D); n=len(a.seeds)
    # 验证集选最强模态
    vtmp={}
    for m in pm: vtmp[m],_,_=train(D,m,a.seeds[0])
    bm=max(pm,key=lambda m:vtmp[m])
    print(f"[control {a.dataset}] best={bm} 训练2组不同init...",flush=True)
    # 两组同模态模型(seed 和 seed+50)
    A={}; B={}; single=[]
    for sd in a.seeds:
        _,pa,yt=train(D,bm,sd); _,pb,_=train(D,bm,sd+50)
        A[sd]=pa; B[sd]=pb
        single.append(f1_score(yt,pa.argmax(1),average="macro",zero_division=0))
    sm_ens=[f1_score(yt,((A[sd]+B[sd])/2).argmax(1),average="macro",zero_division=0) for sd in a.seeds]
    sg=np.array(single); sm=np.array(sm_ens); d=sm-sg
    dm=d.mean(); ci=tcrit(n)*d.std(ddof=1)/np.sqrt(n)
    rng=np.random.RandomState(0); B_=20000
    p=(sum(abs((d*rng.choice([-1,1],size=n)).mean())>=abs(dm) for _ in range(B_))+1)/(B_+1)
    print(f"  best-single {sg.mean():.3f} | same-modal-ens {sm.mean():.3f} | Δ_samemodal {dm:+.3f} (±{ci:.3f}, p={p:.3f})",flush=True)
    print("CTRL_JSON "+json.dumps({"ds":a.dataset,"best":bm,"single":round(float(sg.mean()),3),"samemodal_ens":round(float(sm.mean()),3),"delta_samemodal":round(float(dm),3),"ci":round(float(ci),3),"perm_p":round(float(p),3)}),flush=True)
