#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/build_me.out
#$ -e /home/jianxu/build_me.err
hostname; free -g | head -2
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
python -u build_mimiceye.py
echo BUILD_ME_DONE
