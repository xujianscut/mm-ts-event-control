"""把 Digital Typhoon 事件预测包(dt_et.npz)转成 v2 pipeline 的缓存格式 cache/dt.npz"""
import numpy as np, os
C = "/home/jianxu/v2_cache"
z = np.load(f"{C}/dt_et.npz")
print("源 keys:", z.files)
imgs = z["imgs"]                       # [N,1,96,96] uint8
if imgs.shape[1] == 1:
    imgs = np.repeat(imgs, 3, axis=1)  # -> [N,3,96,96]
times = z["times"].astype(np.float32)  # [N,12,7]
y = z["y"].astype(np.int64)
sp = z["sp"].astype(str)
nC = int(z["nC"])
# 文本模态真实缺失 -> 全零占位(测试门控能否自动关闭无用模态)
Xtxt = np.zeros((len(y), 2), np.float32)
print(f"N={len(y)} imgs={imgs.shape} times={times.shape} nC={nC}")
print("split:", {s: int((sp == s).sum()) for s in np.unique(sp)})
print("正例率:", float(y.mean()))
np.savez_compressed(f"{C}/dt.npz", sp=sp, y=y, nC=np.array(nC),
                    times=times, imgs=imgs, Xtxt=Xtxt)
print(f"saved {C}/dt.npz")
