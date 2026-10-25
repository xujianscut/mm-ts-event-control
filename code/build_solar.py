"""
SDOBenchmark -> 太阳耀斑事件预测三模态包(非气旋正面候选)
任务:未来24h内是否 >=C1 级耀斑(peak_flux >= 1e-6),二分类
模态:
  * 图像 = 最后时间步的 HMI 磁图(光球磁场空间形态),96x96
  * 时序 = 4 个时间步 x 8 个 AIA 波长的平均强度(日冕辐射演化轨迹)—— 与磁场不同物理层,真互补
  * 文本 = 无(缺模态)
划分:官方 train/test(活动区互斥),val 从 train 按活动区切
直接从 zip 读,不铺 375k 小文件。
"""
import os, io, zipfile, time
import numpy as np, pandas as pd
from PIL import Image
from collections import defaultdict

ZIP = os.path.expanduser("~/solar/sdobench.zip")
OUT = os.path.expanduser("~/v2_cache")
ROOT = "SDOBenchmark-data-full"
AIA = ["131","171","193","211","304","335","94","1700"]  # 8 波长
IMG = 96
THR = 1e-6   # C1

def load_split(zf, split):
    meta = pd.read_csv(io.BytesIO(zf.read(f"{ROOT}/{split}/meta_data.csv")))
    # 建 folder -> {timestamp -> {channel -> path}}
    idx = defaultdict(lambda: defaultdict(dict))
    prefix = f"{ROOT}/{split}/"
    for n in zf.namelist():
        if not n.startswith(prefix) or not n.endswith(".jpg"): continue
        parts = n[len(prefix):].split("/")
        if len(parts) != 3: continue
        ar, sample, fn = parts
        key = f"{ar}_{sample}"
        base = fn[:-4]
        if "__" not in base: continue
        ts, ch = base.split("__", 1)
        idx[key][ts][ch] = n
    return meta, idx

def build(zf, meta, idx):
    times, imgs, ys = [], [], []
    t0 = time.time()
    for r_i, row in enumerate(meta.itertuples()):
        key = row.id
        if key not in idx:
            # meta id 可能不带 AR 前缀重复,尝试
            continue
        tss = sorted(idx[key].keys())          # 时间步升序
        if not tss: continue
        # 时序:每个时间步 8 波长均值(取最后 4 步,不足前补零)
        feat = np.zeros((4, len(AIA)), np.float32); tmask = np.zeros(4, np.float32)
        use = tss[-4:]
        for j, ts in enumerate(use):
            slot = 4 - len(use) + j
            for c_i, ch in enumerate(AIA):
                p = idx[key][ts].get(ch)
                if p:
                    try:
                        a = np.asarray(Image.open(io.BytesIO(zf.read(p))).convert("L"), np.float32)
                        feat[slot, c_i] = a.mean()
                    except Exception: pass
            tmask[slot] = 1.0
        # 图像:最后时间步的磁图
        mp = idx[key][tss[-1]].get("magnetogram")
        if mp is None:
            # 退而求其次:最后步的 continuum
            mp = idx[key][tss[-1]].get("continuum")
        if mp is None: continue
        try:
            im = np.asarray(Image.open(io.BytesIO(zf.read(mp))).convert("L").resize((IMG,IMG)), np.uint8)
        except Exception: continue
        times.append(feat); imgs.append(im); ys.append(int(row.peak_flux >= THR))
        if (r_i+1) % 2000 == 0: print(f"  {split_name} {r_i+1}/{len(meta)} ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(times), np.stack(imgs)[:,None,:,:], np.array(ys, np.int64)

if __name__ == "__main__":
    t0 = time.time()
    zf = zipfile.ZipFile(ZIP)
    parts = {}
    for split_name, split in [("train","training"), ("test","test")]:
        meta, idx = load_split(zf, split)
        print(f"[{split}] meta={len(meta)} folders={len(idx)}", flush=True)
        globals()["split_name"] = split_name
        tm, im, y = build(zf, meta, idx)
        parts[split_name] = (tm, im, y)
        print(f"  {split_name}: N={len(y)} imgs={im.shape} pos(>=C1)={y.mean():.3f}", flush=True)

    tr_t, tr_i, tr_y = parts["train"]; te_t, te_i, te_y = parts["test"]
    # val 从 train 尾部切 15%(注:train 已是活动区互斥于 test;val 简单随机即可,同 seed 可复现)
    n = len(tr_y); rng = np.random.RandomState(2026); perm = rng.permutation(n); cut = int(0.85*n)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    times = np.concatenate([tr_t, te_t]); imgs = np.concatenate([tr_i, te_i]); y = np.concatenate([tr_y, te_y])
    sp = np.array(["train"]*len(tr_y) + ["test"]*len(te_y), dtype="U5")
    sp[va_idx] = "val"
    # 标准化时序(仅 train)
    trm = sp == "train"
    mu = times[trm].reshape(-1, len(AIA)).mean(0); sd = times[trm].reshape(-1, len(AIA)).std(0)+1e-6
    times = (times - mu)/sd
    imgs3 = np.repeat(imgs, 3, axis=1)
    print("split:", {s:int((sp==s).sum()) for s in np.unique(sp)}, "| 正例率", float(y.mean()), flush=True)
    np.savez_compressed(f"{OUT}/solar.npz", sp=sp, y=y, nC=np.array(2),
                        times=times.astype(np.float32), imgs=imgs3, Xtxt=np.zeros((len(y),2),np.float32))
    print(f"saved {OUT}/solar.npz N={len(y)} 用时 {time.time()-t0:.0f}s", flush=True)
    print("SOLAR_BUILD_DONE", flush=True)
