"""对 genuine excess = macroF1(ensemble) - macroF1(same-modality control) 直接做
   group-clustered paired bootstrap(回应 reviewer:pboot 必须检验 excess 而非 raw Δens)。
   --enc weak|strong。输出 EXBOOT_JSON。"""
import sys,os,argparse,warnings,json
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from train_v2_raiden import prepare, TimeEnc, ImgEnc, DEV
def macro(y,p): return f1_score(y,p,average="macro",zero_division=0)

# --- strong encoders (same as unified_strong) ---
class StrongTimeEnc(nn.Module):
    def __init__(s,C,h=256):
        super().__init__(); s.gru=nn.GRU(C,h,2,batch_first=True,bidirectional=True,dropout=0.1); s.out=2*h
    def forward(s,x): return s.gru(x)[0].mean(1)
class ResBlock(nn.Module):
    def __init__(s,ci,co,st=1):
        super().__init__(); s.c1=nn.Conv2d(ci,co,3,st,1,bias=False); s.b1=nn.BatchNorm2d(co)
        s.c2=nn.Conv2d(co,co,3,1,1,bias=False); s.b2=nn.BatchNorm2d(co)
        s.sc=nn.Sequential() if (st==1 and ci==co) else nn.Sequential(nn.Conv2d(ci,co,1,st,bias=False),nn.BatchNorm2d(co))
    def forward(s,x): h=F.relu(s.b1(s.c1(x))); h=s.b2(s.c2(h)); return F.relu(h+s.sc(x))
class StrongImgEnc(nn.Module):
    def __init__(s):
        super().__init__(); s.stem=nn.Sequential(nn.Conv2d(3,32,3,1,1,bias=False),nn.BatchNorm2d(32),nn.ReLU())
        s.l1=ResBlock(32,64,2); s.l2=ResBlock(64,128,2); s.l3=ResBlock(128,256,2); s.l4=ResBlock(256,256,2); s.out=256
    def forward(s,x): h=s.l4(s.l3(s.l2(s.l1(s.stem(x))))); return F.adaptive_avg_pool2d(h,1).flatten(1)

class Single(nn.Module):
    def __init__(s,m,C,nC,strong):
        super().__init__()
        if strong: e=StrongTimeEnc(C) if m=="time" else StrongImgEnc()
        else: e=TimeEnc(C) if m=="time" else ImgEnc()
        s.e=e; s.m=m; s.h=nn.Sequential(nn.LayerNorm(e.out),nn.Linear(e.out,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nC))
    def forward(s,b): return s.h(s.e(b[s.m]))

def train(D,m,seed,strong,epochs=None,bs=128,patience=None):
    epochs=epochs or (50 if strong else 60); patience=patience or (10 if strong else 12)
    torch.manual_seed(seed); np.random.seed(seed); sp=D["sp"]; nC=int(D["nC"])
    tr=np.where(np.isin(sp,["train","training"]))[0]; va=np.where(sp=="val")[0]; te=np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr=tr[c:],tr[:c]
    y=torch.tensor(D["y"]); ti=torch.tensor(D["times"]); im=torch.tensor(D["imgs"]).float()/255.; C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Single(m,C,nC,strong).to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
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
    net.load_state_dict(stt); return best, prob(te)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--enc",default="weak")
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(2026,2036))); a=ap.parse_args()
    strong=(a.enc=="strong"); D=prepare(a.dataset); pm=["time","image"]; n=len(a.seeds); k=len(pm)
    te=np.where(D["sp"]=="test")[0]; yte=D["y"][te]
    gidf=f"/home/jianxu/gid_{a.dataset}.npy"; gid=np.load(gidf,allow_pickle=True).astype(str)[te] if os.path.exists(gidf) else None
    print(f"[exboot {a.dataset} enc={a.enc}] testN={len(te)} grouped={gid is not None}",flush=True)
    sv={m:[] for m in pm}; tp={m:[] for m in pm}
    for sd in a.seeds:
        for m in pm:
            v,p=train(D,m,sd,strong); sv[m].append(v); tp[m].append(p)
    bestmod=max(pm,key=lambda m: np.mean([macro(yte,tp[m][i].argmax(1)) for i in range(n)]))  # test-max
    # seed-avg ensemble & control(k copies of best) probas
    ens=np.mean([np.mean(tp[m],0) for m in pm],0)
    ctrl_probs=[np.mean(tp[bestmod],0)]
    for j in range(k-1):
        cp=[train(D,bestmod,sd+1000*(j+1),strong)[1] for sd in a.seeds]; ctrl_probs.append(np.mean(cp,0))
    ctrl=np.mean(ctrl_probs,0)
    me=ens.argmax(1); mc=ctrl.argmax(1)
    exc_point=macro(yte,me)-macro(yte,mc)
    # paired group bootstrap on EXCESS
    rng=np.random.RandomState(1); diffs=[]
    if gid is not None:
        gs=np.array(sorted(set(gid))); gm={g:np.where(gid==g)[0] for g in gs}
        for _ in range(3000):
            ch=rng.choice(gs,len(gs),replace=True); idx=np.concatenate([gm[g] for g in ch])
            diffs.append(macro(yte[idx],me[idx])-macro(yte[idx],mc[idx]))
    else:
        ia=np.arange(len(yte))
        for _ in range(3000):
            idx=rng.choice(ia,len(ia),replace=True); diffs.append(macro(yte[idx],me[idx])-macro(yte[idx],mc[idx]))
    diffs=np.sort(diffs); lo,hi=float(diffs[74]),float(diffs[2924]); p=float((diffs<=0).mean())
    res={"ds":a.dataset,"enc":a.enc,"best":bestmod,"grouped":gid is not None,
         "excess_point":round(exc_point,3),"excess_boot_ci":[round(lo,3),round(hi,3)],"excess_boot_p":round(p,4)}
    print("EXBOOT_JSON "+json.dumps(res),flush=True); print(f"EXBOOT_DONE_{a.dataset}_{a.enc}",flush=True)
