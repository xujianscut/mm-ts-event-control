#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/base_extra.out
#$ -e /home/jianxu/base_extra.err
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
cd ~; export PYTHONWARNINGS=ignore
for ds in dtlf hursat solar mimiceye; do
  echo "######## ShaSpec: $ds ########"
  python -u shaspec_raiden.py --dataset $ds --seeds 2026 2027 2028 2029 2030
done
echo "######## TimeCAP-equiv(TS+text): mimiceye ########"
python -u tstext_raiden.py --dataset mimiceye --seeds 2026 2027 2028 2029 2030
echo BASE_EXTRA_DONE
