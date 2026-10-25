#!/bin/bash
cd /home/jianxu/nbp_hd
mkdir -p jobs
NDP="bridge=false,resample=5,timesteps=100"
CTX="bridge=true,anchor=context,gamma_mode=linear,gamma_scale=1.0,anchor_loss_scale=1.0,resample=5,timesteps=100"
sub(){  # $1=dim $2=seed $3=args $4=expid
cat > jobs/job_$4.sh <<EOF
#!/bin/bash
cd /home/jianxu/nbp_hd/flowNP
singularity exec --nv --bind /home/jianxu:/home/jianxu /lustre/tmp_work/pytorch.sif \
  python gp_hd.py --model=nbpd --model_args "$3" --dim_x=$1 --kernel=rbf \
  --expname=hd --expid=$4 --train_seed=$2 --eval_seed=$2 \
  --num_steps=100000 --print_freq=50000 --eval_freq=100000 --save_freq=100000 \
  --eval_num_batches=2 --eval_batch_size=100 --eval_num_samples=30
EOF
chmod +x jobs/job_$4.sh
rm -rf flowNP/output/results/gp/hd/rbf/nbpd/$4
qsub -jc gtn-container_g1 -ac d=none -N $4 -o /home/jianxu/nbp_hd/$4.out -j y jobs/job_$4.sh
}
# D=3 extra seeds (strengthen the key result)
for S in 2 3 4; do sub 3 $S "$NDP" ndp_d3_s$S; sub 3 $S "$CTX" ctx_d3_s$S; done
# D=4 and D=5
for D in 4 5; do for S in 0 1; do sub $D $S "$NDP" ndp_d${D}_s$S; sub $D $S "$CTX" ctx_d${D}_s$S; done; done
echo "submitted"; qstat -u jianxu | tail -20
