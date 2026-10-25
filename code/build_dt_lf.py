"""
Digital Typhoon LANDFALL -> 事件预测数据包(对照实验)(时序 + 图像,无文本 = 缺模态案例)

任务:温带转换预测(ET) —— 给定过去 12h 的 best-track 数值 + 当前红外云图,
      预测未来 24h 内是否转为温带气旋(grade==6)。
设计要点:
  * 标签在未来 -> 无泄漏(当前 grade/wind 直接决定当前 grade,不能当输入标签)
  * 只保留 t 时刻 grade in {2,3,4,5} 的样本(已是 6 的排除)
  * 严格按【序列】划分 0.7/0.15/0.15 —— 同一台风的相邻帧几乎相同,随机划分会泄漏
  * 每 3 小时采样一次(逐小时帧高度冗余)
输出:cache/dt_et.npz  (times, tmask, imgs uint8 [N,1,96,96], y, sp, nC)
"""
import os, sys, time
import numpy as np, pandas as pd, h5py
from PIL import Image
from multiprocessing import Pool

ROOT = os.path.expanduser("~/digital_typhoon/WP")
OUT = os.path.expanduser("~/v2_cache")
os.makedirs(OUT, exist_ok=True)

K_HIST = 12       # 过去 12 步(小时)
HORIZON = 24      # 预测未来 24h
STRIDE = 3        # 每 3 小时采一个样本
IMG = 96
FEATS = ["pressure", "wind", "lat", "lng", "dir50", "long50", "short50"]


def build_index():
    d = pd.read_csv(f"{ROOT}/aux_data.csv")
    d = d.sort_values(["id", "year", "month", "day", "hour"]).reset_index(drop=True)
    rows = []
    for sid, g in d.groupby("id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        grade = g["grade"].to_numpy()
        lf = g["landfall"].to_numpy()
        for t in range(K_HIST, n - HORIZON, STRIDE):
            if lf[t] > 0:                        # 当前已在登陆的排除
                continue
            label = int((lf[t + 1: t + 1 + HORIZON] > 0).any())
            rows.append((str(sid), t, g.loc[t, "image_path"], label))
    return d, pd.DataFrame(rows, columns=["id", "t", "image_path", "y"])


def load_img(args):
    sid, path = args
    try:
        with h5py.File(f"{ROOT}/image/{sid}/{path}", "r") as h:
            a = np.array(h["Infrared"], dtype=np.float32)
        # 亮温 ~[180,300]K -> 归一到 0-255
        a = np.clip((a - 180.0) / (300.0 - 180.0), 0, 1) * 255.0
        im = Image.fromarray(a.astype(np.uint8)).resize((IMG, IMG), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return np.zeros((IMG, IMG), np.uint8)


if __name__ == "__main__":
    t0 = time.time()
    d, idx = build_index()
    print(f"候选样本: {len(idx)}  序列数: {idx.id.nunique()}  正例率: {idx.y.mean():.4f}", flush=True)

    # 按序列划分(防泄漏),用序列号 hash 保证可复现
    seqs = sorted(idx.id.unique())
    rng = np.random.RandomState(2026); rng.shuffle(seqs)
    n = len(seqs); n_tr, n_va = int(.7 * n), int(.15 * n)
    split_of = {}
    for i, s in enumerate(seqs):
        split_of[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    idx["sp"] = idx.id.map(split_of)
    print("按序列划分:", idx.sp.value_counts().to_dict(), flush=True)

    # 时序特征:过去 K_HIST 步
    dd = d.set_index(["id", d.groupby("id").cumcount()])
    times = np.zeros((len(idx), K_HIST, len(FEATS)), np.float32)
    tmask = np.ones((len(idx), K_HIST), np.float32)
    for i, (sid, t) in enumerate(zip(idx.id.values, idx.t.values)):
        try:
            seg = dd.loc[(int(sid), slice(t - K_HIST, t - 1)), FEATS].to_numpy(np.float32)
            if len(seg) == K_HIST:
                times[i] = np.nan_to_num(seg)
            else:
                times[i, -len(seg):] = np.nan_to_num(seg)
        except Exception:
            tmask[i] = 0
        if (i + 1) % 20000 == 0:
            print(f"  时序 {i+1}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)

    # 标准化(仅用 train)
    tr = (idx.sp == "train").to_numpy()
    mu = times[tr].reshape(-1, len(FEATS)).mean(0)
    sd = times[tr].reshape(-1, len(FEATS)).std(0) + 1e-6
    times = ((times - mu) / sd) * tmask[..., None]

    # 图像(多进程)
    print("开始读图像...", flush=True)
    with Pool(16) as p:
        imgs = p.map(load_img, list(zip(idx.id.values, idx.image_path.values)), chunksize=64)
    imgs = np.stack(imgs)[:, None, :, :]     # [N,1,96,96]
    print(f"图像完成 {imgs.shape} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(f"{OUT}/dt_lf_raw.npz",
                        times=times, tmask=tmask, imgs=imgs,
                        y=idx.y.to_numpy(np.int64), sp=idx.sp.to_numpy().astype("U5"),
                        nC=np.array(2))
    print(f"saved {OUT}/dt_et.npz  N={len(idx)}  正例={idx.y.sum()} ({idx.y.mean():.3f})  用时 {time.time()-t0:.0f}s")
