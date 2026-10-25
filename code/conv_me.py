"""mimiceye_raw.npz -> v2 缓存格式 cache/mimiceye.npz(文本 TF-IDF,图像 1->3 通道)"""
import numpy as np, os
from sklearn.feature_extraction.text import TfidfVectorizer
C = os.path.expanduser("~/v2_cache")
z = np.load(f"{C}/mimiceye_raw.npz", allow_pickle=True)
sp = z["sp"].astype(str); y = z["y"]; times = z["times"].astype(np.float32)
imgs = z["imgs"]
if imgs.shape[1] == 1: imgs = np.repeat(imgs, 3, axis=1)
texts = [str(t) for t in z["texts"]]
tr = np.isin(sp, ["train"])
vec = TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1,2), stop_words="english")
vec.fit([texts[i] for i in np.where(tr)[0]])
Xtxt = vec.transform(texts).toarray().astype(np.float32)
print(f"N={len(y)} imgs={imgs.shape} times={times.shape} Xtxt={Xtxt.shape}")
print("split:", {s:int((sp==s).sum()) for s in np.unique(sp)}, "| 正例率", float(y.mean()))
print("文本样例(前120字):", texts[0][:120].replace("\n"," "))
np.savez_compressed(f"{C}/mimiceye.npz", sp=sp, y=y, nC=np.array(2), times=times, imgs=imgs, Xtxt=Xtxt)
print(f"saved {C}/mimiceye.npz")
