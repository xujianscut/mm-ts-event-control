"""复现 build_hursat 的 recs->rows 逻辑恢复每样本 SID(storm)分组,与缓存 y 对齐校验后保存。"""
import numpy as np, pandas as pd, glob, time
import build_hursat as b
from multiprocessing import Pool
t0=time.time()
tgzs=sorted(glob.glob(f"{b.RAW}/*.tar.gz"))
with Pool(16) as p: res=p.map(b.one_storm, tgzs, chunksize=2)
recs=[x for r in res for x in r]
df=pd.DataFrame([(r[0],r[1]) for r in recs],columns=["SID","ts"]); df["idx"]=np.arange(len(df))
ib=pd.read_csv(b.IBT,low_memory=False,skiprows=[1])[["SID","ISO_TIME","NATURE"]]
ib["ts"]=pd.to_datetime(ib.ISO_TIME,errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
df["ts"]=pd.to_datetime(df.ts,errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
df=df.merge(ib[["SID","ts","NATURE"]],on=["SID","ts"],how="left")
rows=[]
for sid,g in df.sort_values(["SID","ts"]).groupby("SID"):
    g=g.reset_index(drop=True); nat=g.NATURE.fillna("NR").to_numpy(); n=len(g)
    for t in range(b.K_HIST,n-b.HORIZON):
        if nat[t] not in ("TS","DS"): continue
        label=int((nat[t+1:t+1+b.HORIZON]=="ET").any())
        rows.append((sid,label))
gid=np.array([r[0] for r in rows]); yv=np.array([r[1] for r in rows],int)
z=np.load("/home/jianxu/v2_cache/hursat.npz",allow_pickle=True); yc=z["y"].astype(int); spc=z["sp"].astype(str)
ok=(len(yv)==len(yc)) and bool((yv==yc).all())
print(f"[hursat gid] cacheN={len(yc)} rebuiltN={len(yv)} y-aligned={ok} nStorms={len(set(gid))} ({time.time()-t0:.0f}s)",flush=True)
if ok:
    np.save("/home/jianxu/gid_hursat.npy",gid); te=spc=="test"
    print(f"  saved gid_hursat.npy testN={te.sum()} testStorms={len(set(gid[te]))}",flush=True)
print("GIDHURSAT_DONE",flush=True)
