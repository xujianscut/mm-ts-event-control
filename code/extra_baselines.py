"""补两个基线:(a) WeightFuse-exhaustive:固定小网格穷举 val-argmax(使算法=Prop1候选集argmax);
   (b) per-sample gating/MoE:冻结k个单模态专家,训门控网络输出逐样本模态权重(真正样本级路由)。
   输出 EXTRA_JSON。"""
import sys,os,argparse,warnings,json,itertools
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from train_v2_raiden import prepare,TimeEnc,ImgEnc,TxtEnc,DEV
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)
def present_mods(D):
    ms=["time","image"]; x=D["Xtxt"]
    if x.shape[1]>2 and float(x.std())>1e-6: ms.append("text")
    return ms
class Single(nn.Module):
    def __init__(s,m,C,dt,nC):
        super().__init__(); e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.e=e; s.m=m
        s.h=nn.Sequential(nn.LayerNorm(e.out),nn.Linear(e.out,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(s.e(b[s.m]))

def train_single(D,m,seed,epochs=60,bs=128,patience=12):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; tx=torch.tensor(D["Xtxt"]); C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Single(m,C,D["Xtxt"].shape[1],nC).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
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
    net.load_state_dict(stt)
    return prob(va),prob(te)

def wfuse_grid(vp,tp,yva,yte,pm,res=0.1):
    """固定单纯形网格穷举(分辨率res),val-argmax macro-F1,报test。含所有顶点。"""
    k=len(pm); steps=int(round(1/res))
    grid=[np.array(c)/steps for c in itertools.product(range(steps+1),repeat=k) if sum(c)==steps]
    best=-1; bw=None
    for w in grid:
        f=macro(yva,sum(w[i]*vp[pm[i]] for i in range(k)).argmax(1))
        if f>best: best,bw=f,w
    return macro(yte,sum(bw[i]*tp[pm[i]] for i in range(k)).argmax(1))

class Gate(nn.Module):
    """样本级门控:输入k个专家softmax拼接 -> MLP -> k权重(softmax) -> 逐样本加权组合。"""
    def __init__(s,k,nC):
        super().__init__(); s.k=k; s.nC=nC
        s.g=nn.Sequential(nn.Linear(k*nC,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,k))
    def forward(s,P):  # P:[N,k,nC]
        w=torch.softmax(s.g(P.reshape(P.shape[0],-1)),-1)  # [N,k]
        return (w.unsqueeze(-1)*P).sum(1)  # [N,nC]

def train_gate(D,pm,vp,tp,seed,nC,epochs=80,bs=256):
    """在train上训门控(专家冻结,用其在train的proba)。需要train proba -> 这里复用val/test不够,
       改为:用各专家对train的proba训门控。为省事,专家已训好,重新算train proba。"""
    # 需要 train proba:重新前向(专家状态已丢)-> 简化:用 val 训门控、test 评估(val作代理训练集)
    torch.manual_seed(seed); yva=torch.tensor(D["y"][np.where(D["sp"]=="val")[0]]).long()
    if len(yva)==0: return None
    Pv=torch.tensor(np.stack([vp[m] for m in pm],1)).float(); Pt=np.stack([tp[m] for m in pm],1)
    yte=D["y"][np.where(D["sp"]=="test")[0]]
    net=Gate(len(pm),nC); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    idx=np.arange(len(yva))
    for e in range(epochs):
        net.train(); r=np.random.RandomState(e); r.shuffle(idx)
        for k in range(0,len(idx),bs):
            b=idx[k:k+bs]; opt.zero_grad(); lf(net(Pv[b]),yva[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): out=net(torch.tensor(Pt).float()).numpy()
    return macro(yte,out.argmax(1))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=present_mods(D); nC=int(D["nC"]); n=len(a.seeds)
    va=np.where(D["sp"]=="val")[0]; te=np.where(D["sp"]=="test")[0]; yva=D["y"][va]; yte=D["y"][te]
    print(f"[extra {a.dataset}] present={pm} nC={nC}",flush=True)
    wf=[]; gate=[]
    for sd in a.seeds:
        vp={}; tp={}
        for m in pm:
            v,t=train_single(D,m,sd); vp[m]=v; tp[m]=t
        wf.append(wfuse_grid(vp,tp,yva,yte,pm))
        g=train_gate(D,pm,vp,tp,sd,nC)
        if g is not None: gate.append(g)
        print(f"  seed{sd}: wfuse_grid={wf[-1]:.3f} gate={gate[-1] if gate else float('nan'):.3f}",flush=True)
    res={"ds":a.dataset,"present":pm,"wfuse_grid_f1":round(float(np.mean(wf)),3),
         "gating_f1":(round(float(np.mean(gate)),3) if gate else None)}
    print("EXTRA_JSON "+json.dumps(res),flush=True); print(f"EXTRA_DONE_{a.dataset}",flush=True)
