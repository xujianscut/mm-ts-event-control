#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/a2_hursat.out
#$ -e /home/jianxu/a2_hursat.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u audit2.py --dataset hursat --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035
echo A2_DONE_hursat
