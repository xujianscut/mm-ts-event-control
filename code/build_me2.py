"""
MIMIC-Eye -> 真三模态事件预测包

任务:ED disposition 预测(ADMITTED=1 vs HOME=0)
  * 时序:ED/vitalsign.csv 该 stay 的生命体征序列(temperature/heartrate/resprate/o2sat/sbp/dbp)
  * 图像:CXR-JPG/s{study}/{dicom}.jpg
  * 文本:CXR-DICOM/s{study}.txt  <- 真实放射报告全文
  * 链接:cxr_meta.csv 自带 stay_id(MIMIC-Eye 已做时间窗对齐),取非空的
设计:
  * 绝不用 cxr_chexpert 标签(那是从报告 NLP 抽的 -> 文本模态会泄漏)
  * disposition 由医生根据整体临床决定,与报告/影像独立
  * 按【患者】划分 0.7/0.15/0.15 防泄漏
输出:~/v2_cache/mimiceye_raw.npz
"""
import os, glob, time
import numpy as np, pandas as pd
from PIL import Image
from multiprocessing import Pool

ROOT = os.path.expanduser("~/mimic_eye/mimic-eye")
OUT = os.path.expanduser("~/v2_cache")
os.makedirs(OUT, exist_ok=True)
K = 16                       # 最多 16 个生命体征时间点
VIT = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
IMG = 96


def one_patient(pdir):
    out = []
    try:
        meta = pd.read_csv(f"{pdir}/CXR-JPG/cxr_meta.csv")
        if "stay_id" not in meta.columns:
            return out
        meta = meta[meta.stay_id.notna()]
        if not len(meta):
            return out
        eds = pd.read_csv(f"{pdir}/ED/edstays.csv")
        vit = pd.read_csv(f"{pdir}/ED/vitalsign.csv")
    except Exception:
        return out
    pid = os.path.basename(pdir)
    for _, r in meta.iterrows():
        sid = int(r.stay_id); study = int(r.study_id)
        e = eds[eds.stay_id == sid]
        if not len(e):
            continue
        disp = str(e.iloc[0].disposition)
        if disp not in ("ADMITTED", "HOME"):
            continue
        txtf = f"{pdir}/CXR-DICOM/s{study}.txt"
        imgf = f"{pdir}/CXR-JPG/s{study}/{r.dicom_id}.jpg"
        if not (os.path.exists(txtf) and os.path.exists(imgf)):
            continue
        v = vit[vit.stay_id == sid].sort_values("charttime")
        arr = np.zeros((K, len(VIT)), np.float32); m = np.zeros((K,), np.float32)
        if len(v):
            a = v[VIT].to_numpy(np.float32)[:K]
            a = np.nan_to_num(a, nan=0.0)
            arr[:len(a)] = a; m[:len(a)] = 1.0
        try:
            txt = open(txtf, errors="ignore").read()
        except Exception:
            txt = ""
        out.append((pid, imgf, txt, arr, m, int(disp == "ADMITTED")))
    return out


def load_img(f):
    try:
        return np.asarray(Image.open(f).convert("L").resize((IMG, IMG)), np.uint8)
    except Exception:
        return np.zeros((IMG, IMG), np.uint8)


if __name__ == "__main__":
    t0 = time.time()
    pdirs = sorted(glob.glob(f"{ROOT}/patient_*"))
    print(f"patients: {len(pdirs)}", flush=True)
    with Pool(16) as p:
        res = p.map(one_patient, pdirs, chunksize=8)
    rows = [x for r in res for x in r]
    print(f"三模态样本: {len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    if not rows:
        raise SystemExit("没有样本")

    pids = np.array([r[0] for r in rows])
    imgfs = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    times = np.stack([r[3] for r in rows])
    tmask = np.stack([r[4] for r in rows])
    y = np.array([r[5] for r in rows], np.int64)
    print(f"正例(ADMITTED)率: {y.mean():.3f}", flush=True)

    # 按患者划分
    up = sorted(set(pids.tolist()))
    rng = np.random.RandomState(2026); rng.shuffle(up)
    n = len(up); ntr, nva = int(.7 * n), int(.15 * n)
    sp_of = {p: ("train" if i < ntr else ("val" if i < ntr + nva else "test")) for i, p in enumerate(up)}
    sp = np.array([sp_of[p] for p in pids])
    print("患者划分:", {s: int((sp == s).sum()) for s in np.unique(sp)}, flush=True)

    # 标准化时序(仅 train)
    tr = sp == "train"
    mu = times[tr].reshape(-1, len(VIT)).mean(0); sd = times[tr].reshape(-1, len(VIT)).std(0) + 1e-6
    times = ((times - mu) / sd) * tmask[..., None]

    # 图像
    with Pool(16) as p:
        imgs = p.map(load_img, imgfs, chunksize=32)
    imgs = np.stack(imgs)[:, None, :, :]
    print(f"图像: {imgs.shape} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(f"{OUT}/mimiceye_raw.npz", pids=pids, times=times, tmask=tmask, imgs=imgs,
                        y=y, sp=sp.astype("U5"), nC=np.array(2),
                        texts=np.array(texts, dtype=object), allow_pickle=True)
    print(f"saved {OUT}/mimiceye_raw.npz  N={len(y)}  用时 {time.time()-t0:.0f}s")
