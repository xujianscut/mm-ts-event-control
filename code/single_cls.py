"""专门训练的单模态基线(公平的 best-single)—— 每个模态一个独立模型,全部损失都在它自己头上。"""
import sys, os, argparse, warnings, statistics as st
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from train_v2_raiden import prepare, TimeEnc, ImgEnc, TxtEnc, DEV

class Single(nn.Module):
    def __init__(s, mod, C, dtxt, nC):
        super().__init__()
        s.mod = mod
        s.enc = {"time": lambda: TimeEnc(C), "image": ImgEnc, "text": lambda: TxtEnc(dtxt)}[mod]()
        s.head = nn.Sequential(nn.LayerNorm(s.enc.out), nn.Linear(s.enc.out, 128), nn.ReLU(),
                               nn.Dropout(0.3), nn.Linear(128, nC))
    def forward(s, b): return s.head(s.enc(b[s.mod]))

def run(D, mod, seed, epochs=60, bs=128, patience=12):
    torch.manual_seed(seed); np.random.seed(seed)
    sp = D["sp"]; nC = int(D["nC"])
    tr = np.where(np.isin(sp, ["train","training"]))[0]; va = np.where(sp=="val")[0]; te = np.where(sp=="test")[0]
    if len(va)==0:
        r=np.random.RandomState(seed); r.shuffle(tr); c=int(.85*len(tr)); va,tr = tr[c:],tr[:c]
    y=torch.tensor(D["y"]); times=torch.tensor(D["times"]); imgs=torch.tensor(D["imgs"]).float()/255.; txt=torch.tensor(D["Xtxt"])
    C=D["times"].shape[-1]
    cw=np.bincount(D["y"][tr],minlength=nC).astype(np.float32); cw=cw.sum()/(cw+1e-6); cw/=cw.mean()
    net=Single(mod,C,D["Xtxt"].shape[1],nC).to(DEV)
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    lf=nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEV))
    def bat(ix): return {"time":times[ix].to(DEV),"image":imgs[ix].to(DEV),"text":txt[ix].to(DEV)}, y[ix].to(DEV)
    def ev(ix):
        net.eval(); ps,pr=[],[]
        with torch.no_grad():
            for k in range(0,len(ix),512):
                b,_=bat(ix[k:k+512]); o=net(b)
                ps.append(o.argmax(-1).cpu().numpy()); pr.append(torch.softmax(o,-1).cpu().numpy())
        p=np.concatenate(ps); pr=np.concatenate(pr); yy=D["y"][ix]
        try: auc=roc_auc_score(yy,pr[:,1]) if nC==2 else roc_auc_score(yy,pr,multi_class="ovr")
        except Exception: auc=float("nan")
        return f1_score(yy,p,average="macro",zero_division=0), accuracy_score(yy,p), auc
    best,bstate,bad=-1,None,0
    for ep in range(epochs):
        net.train(); rr=np.random.RandomState(ep); o=tr.copy(); rr.shuffle(o)
        for k in range(0,len(o),bs):
            b,yb=bat(o[k:k+bs]); opt.zero_grad(); lf(net(b),yb).backward(); opt.step()
        sch.step(); vf=ev(va)[0]
        if vf>best: best,bstate,bad=vf,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(bstate); return ev(te)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--mods",nargs="+",default=["time","image","text"])
    ap.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]); a=ap.parse_args()
    D=prepare(a.dataset)
    for mod in a.mods:
        fs=[]
        for sd in a.seeds:
            f,ac,au=run(D,mod,sd); fs.append(f)
            print(f"  [{mod}] seed{sd}: macroF1={f:.3f} acc={ac:.3f} auc={au:.3f}",flush=True)
        print(f"  >> DEDICATED {mod}-only macroF1 {st.mean(fs):.3f}±{(st.pstdev(fs) if len(fs)>1 else 0):.3f}",flush=True)
