#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/dt_single.out
#$ -e /home/jianxu/dt_single.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
echo "######## DEDICATED single-modality on DT ########"
python -u single_cls.py --dataset dt --mods image time --seeds 2026 2027 2028
echo DT_SINGLE_DONE
