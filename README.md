# When Does Multimodality Help Time-Series Event Prediction? A Same-Modality Control

论文的实验代码、运行日志和预处理数据。实验于 2026-07-16 至 07-21 在 RIKEN Raiden 集群上运行，打包于 2026-10-25。

## 下载

**完整数据包（约 2.1 GB）：DOWNLOAD_LINK_PLACEHOLDER**

数据包内容与本仓库相同，额外包含 `data/processed/` 下 7 个预处理好的 `.npz`（GitHub 不允许单个文件超过 100 MB，所以没有放进 git）。下载解压后用 `data/processed/MD5SUMS` 校验：

```bash
cd data/processed && md5sum -c MD5SUMS      # macOS: 逐个 md5 -r 对照
```

## 目录结构

```
code/            全部 Python 脚本 + UGE 作业脚本（*.sh）
logs/raiden/     Raiden 上每个作业的 stdout（*.out）及构建日志
logs/autodl/     早期在 AutoDL 上跑的 weather1 / weather2 / traffic / medical 等日志（只有这些文本）
logs/early/      07-16～07-18 早期 Raiden 输出的本地副本（与 logs/raiden 部分重复）
data/processed/  实验直接读取的 .npz 缓存（原 ~/v2_cache；仅在上面的数据包里）
data/gid/        聚类 bootstrap 用的分组 ID（gid_<ds>.npy，按风暴 / 活动区分组）
raw_manifests/   重建原始数据用的清单和下载脚本、IBTrACS 快照
```

## 处理后数据（data/processed）

拿到这些就可以直接重跑所有实验，不需要下载原始数据。

| 文件 | 数据集 / 任务 | 由谁生成 | 备注 |
|---|---|---|---|
| `dt.npz` | Digital Typhoon，未来 24h 温带转换（DT-ET） | `build_dt.py` → `conv_dt.py` | 实验用 |
| `dt_et.npz` | 同上，转换前的中间格式 | `build_dt.py` | |
| `dtlf.npz` | Digital Typhoon，未来 24h 登陆（DT-Landfall） | `build_dt_lf.py` → `run_lf.sh` 内联转换 | 实验用 |
| `dt_lf_raw.npz` | 同上，中间格式 | `build_dt_lf.py` | |
| `hursat.npz` | HURSAT-B1 + IBTrACS，温带转换（独立数据源复现） | `build_hursat.py` → `run_hursat.sh` 内联转换 | 实验用 |
| `hursat_raw.npz` | 同上，中间格式 | `build_hursat.py` | |
| `solar.npz` | SDOBenchmark，未来 24h ≥C1 耀斑 | `build_solar.py` | 实验用 |
| `mimiceye.npz`、`mimiceye_raw.npz` | MIMIC-Eye，ED disposition（入院 vs 回家） | `build_me2.py` → `conv_me.py` | **不提供**，受 PhysioNet DUA 约束，需自行重建（见下文第 4 节） |

`build_dt.py` 默认输出到 `~/dt_cache/`，而 `conv_dt.py` 读 `~/v2_cache/dt_et.npz`；当时两处的文件 MD5 相同，这里只保留一份。重跑时把它复制到 `~/v2_cache/` 即可。

**MIMIC-Eye 的两个版本**：原实验中 `mimiceye.npz` 由第一版 `build_mimiceye.py` 生成，`mimiceye_raw.npz` 由后来的 `build_me2.py` 重建（多了患者 ID `pids`），之后没有重跑 `conv_me.py`。按数据集名读 `<ds>.npz` 的脚本（`unified2.py` 等）用的是前者；按患者分组 CV 的 `mimic_cv.py`、`cv_me.py` 直接读后者。论文里 MIMIC 的主表数字来自 CV（`mcv.out`）。

完整性：`data/processed/MD5SUMS` 与 Raiden 上的原文件逐一核对一致。

统一格式：`sp`（train/val/test 划分）、`y`、`nC`、`times`（时序）、`imgs`（uint8，[N,3,96,96]）、`Xtxt`（文本特征，无文本的数据集为全零占位）。所有划分都按风暴 / 活动区 / 患者分组切，不是随机切。

## 原始数据下载（共约 155 GB，按需下载）

脚本里的路径都写死为 `~/<目录>`，下载到下面对应位置即可直接运行 `build_*.py`。

### 1. Digital Typhoon（112 GB，许可 CC BY 4.0）

用于 DT-ET 和 DT-Landfall。用的是 torchgeo 在 HuggingFace 上的镜像，只下 WP（西北太平洋）分卷：

```bash
mkdir -p ~/digital_typhoon && cd ~/digital_typhoon
B=https://huggingface.co/datasets/torchgeo/digital_typhoon/resolve/main
wget -c $B/WP.tar.gzaa      # 40 GB
wget -c $B/WP.tar.gzab      # 14 GB
cat WP.tar.gzaa WP.tar.gzab | tar xzf -     # 解出 ~/digital_typhoon/WP/（约 58 GB）
```

（即 `raw_manifests/dt_dl.sh` 和 `dt_ex.sh`。）解压后应有 `WP/aux_data.csv`、`WP/metadata/`、`WP/image/<序列号>/*.h5`，共 1099 个台风序列。两个分卷加解压目录合计约 112 GB，解压完可删分卷。
官方主页：http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ ；论文 Kitamoto et al., NeurIPS 2023 D&B。

### 2. HURSAT-B1 v06 + IBTrACS（32 GB，NOAA 公有领域）

用于 HURSAT-ET。下载 2000～2015 年全部风暴的 tar.gz（1082 个），完整 URL 清单在 `raw_manifests/hursat_urls.txt`：

```bash
mkdir -p ~/hursat/raw && cd ~/hursat/raw
xargs -a /path/to/raw_manifests/hursat_urls.txt -P 8 -I{} wget -q -c -nc {}
```

或者运行 `code/hursat_dl.sh`，它会从 NCEI 目录页重新抓链接：
https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/

IBTrACS：`build_hursat.py` 读 `~/hursat/ibtracs.csv`。**请用本包里的快照**：

```bash
gunzip -c raw_manifests/ibtracs.csv.gz > ~/hursat/ibtracs.csv
```

这是 2026-07-16 下载的全海盆版本（ALL list，v04，含 EP/NA/NI/SA/SI/SP/WP）。IBTrACS 会持续修订历史记录，重新下载的新版本标签可能有细微差异。官方地址：
https://www.ncei.noaa.gov/products/international-best-track-archive （CSV 版 `ibtracs.ALL.list.v04r01.csv`）

`build_hursat.py` 边解包边读 netCDF，不会把 22 万个小文件铺到磁盘。

### 3. SDOBenchmark（3 GB，MIT）

```bash
mkdir -p ~/solar && cd ~/solar
wget -O sdobench.zip https://github.com/i4Ds/SDOBenchmark/archive/data-full.zip
```

`build_solar.py` 直接从 zip 读（zip 内根目录为 `SDOBenchmark-data-full/`），不用解压。使用官方 train/test 划分，val 从 train 按活动区切出。

### 4. MIMIC-Eye（约 10 GB，需 PhysioNet credentialed 权限）

需要先在 PhysioNet 完成 CITI 培训并签署 DUA：
https://physionet.org/content/mimic-eye-multimodal-datasets/1.0.0/

然后把账号写进 `~/.netrc`（`machine physionet.org login <用户名> password <密码>`，权限 600）。患者清单需要登录后从数据集目录索引页生成（原实验为 3192 名患者；清单来自受限目录，本仓库不提供），再定向下载每名患者的 `ED/`、`CXR-JPG/`、`CXR-DICOM/` 三个子目录：

```bash
mkdir -p ~/mimic_eye && cd ~/mimic_eye
wget -q -O idx.html https://physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye/
grep -oE 'patient_[0-9]+' idx.html | sort -u > patients.txt
bash /path/to/code/me_dl.sh
```

之后依次运行 `code/build_me2.py`、`code/conv_me.py` 生成 `~/v2_cache/mimiceye_raw.npz` 和 `mimiceye.npz`。

### 5. weather1 / weather2 / traffic（不在本包内）

这三个数据集以及早期的 medical、stock2 是在 AutoDL 实例（`/root/autodl-tmp/PythonProject_v2/cache/*.npz`）上处理和运行的。该实例已无法连接，**缓存和最终审计（audit2_autodl / control_autodl）的原始输出都没能取回**，只剩 `logs/autodl/` 里的早期文本日志。这三个数据集的最终数字以论文表格为准。

## 代码与日志对应关系

所有主实验都是 10 个 seed（2026～2035）。日志文件名格式为 `<前缀>_<数据集>.out`，数据集取值：`dt`（DT-ET）、`dtlf`（DT-Landfall）、`hursat`、`solar`、`mimiceye`。每个日志末尾有一行 `XXX_JSON {...}`，里面是汇总数字，可以直接解析。

| 前缀 | 脚本 | 内容 | 时间 |
|---|---|---|---|
| `uni_` / `u2_` | `unified.py` / `unified2.py` | **论文主表的统一实验**：单一 RNG 流，独立单模态、FullConcat、ShaSpec*、TimeCAP*、验证集加权 late fusion、uniform ensemble、size-matched 同模态 control、oracle、聚类显著性。`u2` 是终版 | 07-19～07-21 |
| `st_` | `unified_strong.py` | 强编码器稳健性（深残差 CNN + 2 层 BiGRU） | 07-20 |
| `ca_` | `crossattn.py` | 交叉注意力联合融合基线 | 07-20 |
| `bf_` | `boot_full.py` | 对 Δfull 和 Δens 做按风暴聚类的 bootstrap | 07-20 |
| `eb_` | `excess_boot.py` | 对净互补（ensemble − 同模态 control）直接做分组 bootstrap，`--enc weak/strong` | 07-21 |
| `xb_` | `extra_baselines.py` | 补充基线：穷举加权融合、逐样本门控 MoE | 07-21 |
| `mcv.out` | `mimic_cv.py` | MIMIC-Eye 按患者 5 折 CV | 07-21 |
| `a2_` | `audit2.py` | 前一版统一审计（Δfull、Δens、置换检验、图像置换掉幅） | 07-19 |
| `ctrl_` | `control.py` | 前一版同模态对照（2 个 init 的 best-modality 集成） | 07-19 |
| `ens_` / `audit_` | `ens.py` / `audit.py` | 更早的 ensemble 和审计 | 07-18～07-19 |
| `subset.out`、`lf.out`、`solar.out`、`cv.out`、`dt*.out` 等 | `subset_cls.py`、`single_cls.py`、`cv_me.py`、`train_v2_raiden.py` 等 | 07-16～07-18 早期探索（SubsetSelect、门控融合 v2、ShaSpec / TimeCAP 复现） | 早期 |

其他脚本：`make_gid*.py` 生成 `data/gid/`；`gen_jobs.py` 批量生成作业脚本；`shaspec_raiden.py`、`tstext_raiden.py` 分别是 ShaSpec 和 TimeCAP 等价基线的复现；`monotone_cls.py`、`train_v2_raiden.py` 是被放弃的早期方法。

## 运行环境

Raiden：`qsub -jc gtn-container_g1_dev`（A100），conda base，torch 2.5.1+cu121。作业脚本头部有 UGE 参数，所有路径都假设在 `~` 下运行（`cd ~; python -u <script>.py --dataset <ds> --seeds 2026 ... 2035`）。换机器时需要把脚本里的 `/home/jianxu/` 和 `~/v2_cache` 改成本地路径（`grep -n "home/jianxu\|v2_cache" code/*.py`）。

## 数据许可

- 本包中的派生数据来自 Digital Typhoon（CC BY 4.0，Kitamoto et al. 2023）、HURSAT-B1 与 IBTrACS（NOAA，公有领域）、SDOBenchmark（MIT），使用时请引用各数据集的原始出处。
- MIMIC-Eye 派生数据受 PhysioNet Credentialed Health Data License 约束，本仓库和数据包均不包含。
