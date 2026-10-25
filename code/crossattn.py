"""交叉注意力联合融合基线:相同(弱)编码器,融合方式 concat -> transformer自注意力(CLS+模态token)。
   对比 best-single / FullConcat / CrossAttn / Ensemble。若 CrossAttn 也打不过 Ensemble,
   则'联合融合本身'漏互补,不只是concat naive。跑地球物理正例 + weather1(负例对照)。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from train_v2_raiden import prepare,TimeEnc,ImgEnc,TxtEnc,DEV

def present_mods(D):
    ms=["time","image"]; x=D["Xtxt"]
    if x.shape[1]>2 and float(x.std())>1e-6: ms.append("text")
    return ms
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)
def tcrit(n): return {2:12.706,5:2.776,8:2.365,10:2.262}.get(n,1.96)

class Single(nn.Module):
    def __init__(s,m,C,dt,nC):
        super().__init__(); e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.e=e; s.m=m
        s.h=nn.Sequential(nn.LayerNorm(e.out),nn.Linear(e.out,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(s.e(b[s.m]))
class Concat(nn.Module):
    def __init__(s,mods,C,dt,nC):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); d=0
        for m in s.mods:
            e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.enc[m]=e; d+=e.out
        s.h=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(torch.cat([s.enc[m](b[m]) for m in s.mods],-1))
class CrossAttn(nn.Module):
    """每模态嵌入投到 d=128 token,加CLS,2层transformer自注意力,从CLS分类(学习式联合融合)。"""
    def __init__(s,mods,C,dt,nC,d=128):
        super().__init__(); s.mods=list(mods); s.enc=nn.ModuleDict(); s.proj=nn.ModuleDict()
        for m in s.mods:
            e={"time":lambda:TimeEnc(C),"image":ImgEnc,"text":lambda:TxtEnc(dt)}[m](); s.enc[m]=e; s.proj[m]=nn.Linear(e.out,d)
        s.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
        s.typ=nn.Parameter(torch.randn(len(s.mods)+1,d)*0.02)
        layer=nn.TransformerEncoderLayer(d,4,dim_feedforward=256,dropout=0.1,batch_first=True,activation="gelu")
        s.tr=nn.TransformerEncoder(layer,2); s.h=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,nC))
    def forward(s,b):
        toks=[s.cls.expand(b[s.mods[0]].shape[0] if s.mods[0]!="time" else b["time"].shape[0],-1,-1)]
        for m in s.mods: toks.append(s.proj[m](s.enc[m](b[m])).unsqueeze(1))
        x=torch.cat(toks,1)+s.typ.unsqueeze(0)
        return s.h(s.tr(x)[:,0])

def train(D,kind,mods,seed,epochs=60,bs=128,patience=12):
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; tx=torch.tensor(D["Xtxt"]); C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    if kind=="single": net=Single(mods[0],C,D["Xtxt"].shape[1],nC)
    elif kind=="concat": net=Concat(mods,C,D["Xtxt"].shape[1],nC)
    else: net=CrossAttn(mods,C,D["Xtxt"].shape[1],nC)
    net=net.to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
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
    return {"val_f1":macro(D["y"][va],prob(va).argmax(1)),"te_prob":prob(te)}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    D=prepare(a.dataset); pm=present_mods(D); n=len(a.seeds)
    te=np.where(D["sp"]=="test")[0]; yte=D["y"][te]
    print(f"[crossattn {a.dataset}] present={pm} testN={len(te)} seeds={n}",flush=True)
    sv={m:[] for m in pm}; ste={m:[] for m in pm}; fte=[]; cate=[]; ete=[]
    for sd in a.seeds:
        tp={}
        for m in pm:
            r=train(D,"single",[m],sd); sv[m].append(r["val_f1"]); tp[m]=r["te_prob"]; ste[m].append(macro(yte,tp[m].argmax(1)))
        fte.append(macro(yte,train(D,"concat",pm,sd)["te_prob"].argmax(1)))
        cate.append(macro(yte,train(D,"crossattn",pm,sd)["te_prob"].argmax(1)))
        ete.append(macro(yte,np.mean([tp[m] for m in pm],0).argmax(1)))
        print(f"  seed{sd}: full={fte[-1]:.3f} crossattn={cate[-1]:.3f} ens={ete[-1]:.3f}",flush=True)
    bestmod=max(pm,key=lambda m: np.mean(sv[m])); bs=np.array(ste[bestmod])
    def dstat(arr):
        d=arr-bs; dm=d.mean(); ci=tcrit(n)*d.std(ddof=1)/np.sqrt(n)
        rng=np.random.RandomState(0); p=(sum(abs((d*rng.choice([-1,1],size=n)).mean())>=abs(dm) for _ in range(20000))+1)/20001
        return round(float(dm),3),round(float(ci),3),round(float(p),3)
    res={"ds":a.dataset,"best":bestmod,"best_f1":round(float(bs.mean()),3),
         "full_f1":round(float(np.mean(fte)),3),"dfull":dstat(np.array(fte)),
         "crossattn_f1":round(float(np.mean(cate)),3),"dcrossattn":dstat(np.array(cate)),
         "ens_f1":round(float(np.mean(ete)),3),"dens":dstat(np.array(ete))}
    print("CROSSATTN_JSON "+json.dumps(res),flush=True); print(f"CROSSATTN_DONE_{a.dataset}",flush=True)
