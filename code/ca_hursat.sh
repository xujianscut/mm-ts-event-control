#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/ca_hursat.out
#$ -e /home/jianxu/ca_hursat.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u crossattn.py --dataset hursat --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035
echo CA_ALLDONE_hursat
