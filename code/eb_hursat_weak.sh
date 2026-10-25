#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/eb_hursat_weak.out
#$ -e /home/jianxu/eb_hursat_weak.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u excess_boot.py --dataset hursat --enc weak --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035
