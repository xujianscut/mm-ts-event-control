#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/dt_base.out
#$ -e /home/jianxu/dt_base.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
echo "######## ShaSpec on DT ########"
python -u shaspec_raiden.py --dataset dt --seeds 2026 2027 2028
echo "######## TS+Text(TimeCAP-equiv) on DT ########"
python -u tstext_raiden.py --dataset dt --seeds 2026 2027 2028
echo DT_BASE_DONE
