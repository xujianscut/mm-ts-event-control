#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/uni_smoke.out
#$ -e /home/jianxu/uni_smoke.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u unified.py --dataset solar --seeds 2026 2027
echo UNI_SMOKE_DONE
