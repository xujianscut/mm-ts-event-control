#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/me_all.out
#$ -e /home/jianxu/me_all.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
S="2026 2027 2028 2029 2030 2031 2032 2033 2034 2035"
echo "######## DEDICATED singles (10 seeds) ########"
python -u single_cls.py --dataset mimiceye --mods time image text --seeds $S
echo "######## v2 gated fusion (anchor=text, 10 seeds) ########"
python -u train_v2_raiden.py --dataset mimiceye --anchor text --seeds $S
echo "######## v2 gated fusion (anchor=image, 10 seeds) ########"
python -u train_v2_raiden.py --dataset mimiceye --anchor image --seeds $S
echo "######## ShaSpec (10 seeds) ########"
python -u shaspec_raiden.py --dataset mimiceye --seeds $S
echo "######## TS+Text (TimeCAP-equiv, 10 seeds) ########"
python -u tstext_raiden.py --dataset mimiceye --seeds $S
echo ME_ALL_DONE
