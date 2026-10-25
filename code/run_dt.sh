#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/dt_train.out
#$ -e /home/jianxu/dt_train.err
hostname; nvidia-smi -L
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~
export PYTHONWARNINGS=ignore
python -u train_v2_raiden.py --dataset dt --anchor image --seeds 2026 2027 2028
echo DT_TRAIN_DONE
