#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/subset.out
#$ -e /home/jianxu/subset.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
for ds in mimiceye dt; do
  echo "######## SubsetSelect: $ds ########"
  python -u subset_cls.py --dataset $ds --seeds 2026 2027 2028 2029 2030
done
echo SUBSET_DONE
