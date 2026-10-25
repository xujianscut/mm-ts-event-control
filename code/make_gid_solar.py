"""重放 build_solar 的纳入逻辑(不解码图像,只看zip文件列表)恢复 AR 分组 id,与缓存 y 对齐校验。"""
import numpy as np, zipfile, importlib
b=importlib.import_module("build_solar")
zf=zipfile.ZipFile(b.ZIP)
gid=[]; yv=[]
for split_name,split in [("train","training"),("test","test")]:
    meta,idx=b.load_split(zf,split)
    for row in meta.itertuples():
        key=row.id
        if key not in idx: continue
        tss=sorted(idx[key].keys())
        if not tss: continue
        last=idx[key][tss[-1]]
        mp = last.get("magnetogram") or last.get("continuum")
        if mp is None: continue
        gid.append(str(key).split("_")[0]); yv.append(int(row.peak_flux>=b.THR))
gid=np.array(gid); yv=np.array(yv,int)
z=np.load("/home/jianxu/v2_cache/solar.npz",allow_pickle=True); yc=z["y"].astype(int); spc=z["sp"].astype(str)
ok=(len(yv)==len(yc)) and bool((yv==yc).all())
print(f"[solar gid] cacheN={len(yc)} rebuiltN={len(yv)} y-aligned={ok} nAR={len(set(gid))}")
if not ok:
    print("  MISALIGNED"); import sys; sys.exit(1)
np.save("/home/jianxu/gid_solar.npy", gid)
te=spc=="test"; print(f"  saved gid_solar.npy testN={te.sum()} testAR={len(set(gid[te]))}")
