#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/solar.out
#$ -e /home/jianxu/solar.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u build_solar.py
echo "######## SubsetSelect: solar ########"
python -u subset_cls.py --dataset solar --seeds 2026 2027 2028 2029 2030
echo SOLAR_ALL_DONE
