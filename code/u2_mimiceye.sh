#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/u2_mimiceye.out
#$ -e /home/jianxu/u2_mimiceye.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u unified2.py --dataset mimiceye --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035
echo U2_DONE_mimiceye
