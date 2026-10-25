"""重跑 build 的 index 逻辑(不解码图像)恢复 group id,与缓存 y 逐元素对齐校验后保存。"""
import sys, numpy as np, importlib
ds = sys.argv[1]
cache = f"/home/jianxu/v2_cache/{ds}.npz"
z = np.load(cache, allow_pickle=True)
y_cache = z["y"].astype(int)
sp_cache = z["sp"].astype(str)

if ds in ("dt", "dtlf"):
    mod = importlib.import_module("build_dt" if ds == "dt" else "build_dt_lf")
    d, idx = mod.build_index()
    gid = idx.id.values.astype(str)
    y_idx = idx.y.to_numpy(int)
else:
    raise SystemExit(f"no cheap gid path for {ds}")

# 对齐校验
ok = (len(y_idx) == len(y_cache)) and bool((y_idx == y_cache).all())
print(f"[{ds}] cacheN={len(y_cache)} idxN={len(y_idx)} y-aligned={ok} nGroups={len(set(gid))}", flush=True)
if not ok:
    print(f"  !! MISALIGNED, first-mismatch idx:", int(np.argmax(y_idx[:len(y_cache)] != y_cache)) if len(y_idx)==len(y_cache) else "lenDiff", flush=True)
    sys.exit(1)
np.save(f"/home/jianxu/gid_{ds}.npy", gid)
te = sp_cache == "test"
print(f"  saved gid_{ds}.npy  testN={te.sum()} testGroups={len(set(gid[te]))}", flush=True)
