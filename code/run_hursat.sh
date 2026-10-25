#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/hursat_build.out
#$ -e /home/jianxu/hursat_build.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u build_hursat.py
python - <<PY
import numpy as np, os
C=os.path.expanduser("~/v2_cache"); z=np.load(f"{C}/hursat_raw.npz")
imgs=z["imgs"]; imgs=np.repeat(imgs,3,axis=1) if imgs.shape[1]==1 else imgs
np.savez_compressed(f"{C}/hursat.npz", sp=z["sp"].astype(str), y=z["y"], nC=np.array(2),
                    times=z["times"], imgs=imgs, Xtxt=np.zeros((len(z["y"]),2),np.float32))
print("saved hursat.npz N=",len(z["y"]),"pos=",float(z["y"].mean()))
PY
echo "######## SubsetSelect: HURSAT-ET ########"
python -u subset_cls.py --dataset hursat --seeds 2026 2027 2028 2029 2030
echo HURSAT_ALL_DONE
