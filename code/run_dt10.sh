#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/dt10.out
#$ -e /home/jianxu/dt10.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
S="2026 2027 2028 2029 2030 2031 2032 2033 2034 2035"
echo "######## v2 gated fusion (10 seeds) ########"
python -u train_v2_raiden.py --dataset dt --anchor image --seeds $S
echo "######## DEDICATED time-only (10 seeds) ########"
python -u single_cls.py --dataset dt --mods time --seeds $S
echo "######## DEDICATED image-only (10 seeds) ########"
python -u single_cls.py --dataset dt --mods image --seeds $S
echo "######## ShaSpec (10 seeds) ########"
python -u shaspec_raiden.py --dataset dt --seeds $S
echo DT10_DONE
