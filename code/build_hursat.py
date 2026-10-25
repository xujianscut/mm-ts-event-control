"""
HURSAT-B1 + IBTrACS -> 温带转换(ET)事件预测数据包
=> Digital Typhoon ET 任务的【独立复现】(不同数据源/卫星/机构/盆地)

  * 图像:IRWIN 红外亮温 301x301 -> 96x96
  * 时序:过去 12 步(HURSAT 为 3 小时间隔 => 36h)的 [CentLat, CentLon, WindSpd, CentPrs]
  * 标签:未来 24h(=8 步)内 IBTrACS NATURE 是否变为 ET
  * 只取 t 时刻 NATURE ∈ {TS, DS} 的样本(已是 ET 的排除)
  * 按【风暴序列】划分 0.7/0.15/0.15
边解包边读边删,避免 22 万小文件落盘。
"""
import os, glob, re, tarfile, tempfile, shutil, time
import numpy as np, pandas as pd
from PIL import Image
from multiprocessing import Pool
import netCDF4 as nc

RAW = os.path.expanduser("~/hursat/raw")
OUT = os.path.expanduser("~/v2_cache")
IBT = os.path.expanduser("~/hursat/ibtracs.csv")
IMG = 96
K_HIST, HORIZON = 12, 8      # 3小时间隔 -> 12步=36h历史, 8步=24h预测
FEATS = ["CentLat", "CentLon", "WindSpd", "CentPrs"]
FN = re.compile(r"^([0-9A-Z]+)\.([^.]*)\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{4})\.")


def one_storm(tgz):
    """返回该风暴的 [(sid, dt, img96, feats)] 逐时间步(每时间步选一颗卫星)"""
    out = []
    tmp = tempfile.mkdtemp(prefix="hs_")
    try:
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(tmp)
        best = {}   # ts -> (vza, path)
        for f in glob.glob(f"{tmp}/**/*.nc", recursive=True):
            m = FN.match(os.path.basename(f))
            if not m: continue
            sid, _, Y, M, D, HM = m.groups()
            ts = f"{Y}-{M}-{D} {HM[:2]}:{HM[2:]}:00"
            try:
                d = nc.Dataset(f)
                vza = float(np.array(d.variables["VZA"][:]).ravel()[0]) if "VZA" in d.variables else 99.
                d.close()
            except Exception:
                continue
            if ts not in best or vza < best[ts][0]:
                best[ts] = (vza, f, sid)
        for ts, (vza, f, sid) in best.items():
            try:
                d = nc.Dataset(f)
                ir = np.array(d.variables["IRWIN"][:]).squeeze().astype(np.float32)
                fe = [float(np.array(d.variables[k][:]).ravel()[0]) if k in d.variables else 0.0 for k in FEATS]
                d.close()
                if ir.shape != (301, 301): continue
                ir = np.nan_to_num(ir, nan=300.0)
                a = np.clip((ir - 180.0) / 120.0, 0, 1) * 255.0
                im = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((IMG, IMG), Image.BILINEAR), np.uint8)
                out.append((sid, ts, im, np.array(fe, np.float32)))
            except Exception:
                continue
    except Exception:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    tgzs = sorted(glob.glob(f"{RAW}/*.tar.gz"))
    print(f"风暴数: {len(tgzs)}", flush=True)
    with Pool(16) as p:
        res = p.map(one_storm, tgzs, chunksize=2)
    recs = [x for r in res for x in r]
    print(f"读到时间步: {len(recs)}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame([(r[0], r[1]) for r in recs], columns=["SID", "ts"])
    df["idx"] = np.arange(len(df))
    imgs_all = np.stack([r[2] for r in recs])
    feats_all = np.stack([r[3] for r in recs])

    ib = pd.read_csv(IBT, low_memory=False, skiprows=[1])[["SID", "ISO_TIME", "NATURE"]]
    ib["ts"] = pd.to_datetime(ib.ISO_TIME, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["ts"] = pd.to_datetime(df.ts, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.merge(ib[["SID", "ts", "NATURE"]], on=["SID", "ts"], how="left")
    print("NATURE 匹配率:", float(df.NATURE.notna().mean()), flush=True)
    print(df.NATURE.value_counts().head(), flush=True)

    rows = []
    for sid, g in df.sort_values(["SID", "ts"]).groupby("SID"):
        g = g.reset_index(drop=True)
        nat = g.NATURE.fillna("NR").to_numpy(); n = len(g)
        for t in range(K_HIST, n - HORIZON):
            if nat[t] not in ("TS", "DS"): continue
            label = int((nat[t+1:t+1+HORIZON] == "ET").any())
            rows.append((sid, g.loc[t, "idx"], [g.loc[j, "idx"] for j in range(t-K_HIST, t)], label))
    print(f"样本: {len(rows)}  正例率: {np.mean([r[3] for r in rows]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if not rows: raise SystemExit("无样本")

    sids = sorted(set(r[0] for r in rows))
    rng = np.random.RandomState(2026); rng.shuffle(sids)
    n = len(sids); ntr, nva = int(.7*n), int(.15*n)
    sp_of = {s: ("train" if i < ntr else ("val" if i < ntr+nva else "test")) for i, s in enumerate(sids)}
    sp = np.array([sp_of[r[0]] for r in rows])
    y = np.array([r[3] for r in rows], np.int64)
    imgs = imgs_all[[r[1] for r in rows]][:, None, :, :]
    times = np.stack([feats_all[r[2]] for r in rows]).astype(np.float32)
    tr = sp == "train"
    mu = times[tr].reshape(-1, len(FEATS)).mean(0); sd = times[tr].reshape(-1, len(FEATS)).std(0)+1e-6
    times = (times - mu) / sd
    print("划分:", {s: int((sp==s).sum()) for s in np.unique(sp)}, flush=True)
    np.savez_compressed(f"{OUT}/hursat_raw.npz", times=times, imgs=imgs, y=y, sp=sp.astype("U5"), nC=np.array(2))
    print(f"saved {OUT}/hursat_raw.npz N={len(y)} 用时 {time.time()-t0:.0f}s", flush=True)
    print("HURSAT_BUILD_DONE", flush=True)
