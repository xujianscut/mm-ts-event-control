#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/audit_solar.out
#$ -e /home/jianxu/audit_solar.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u audit.py --dataset solar --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035
echo AUDIT_DONE_solar
