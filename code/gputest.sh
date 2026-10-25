#!/bin/bash
#$ -jc gtn-container_g1_dev
#$ -ac d=none
#$ -cwd
#$ -o /home/jianxu/gputest.out
#$ -e /home/jianxu/gputest.err
hostname
nvidia-smi -L
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
python -c "import torch; print(\"torch\", torch.__version__, \"| cuda:\", torch.cuda.is_available(), \"|\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\")"
