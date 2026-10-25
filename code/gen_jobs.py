import os
base = "/home/jianxu/nbp_hd"
flowdir = base + "/flowNP"
sif = "/lustre/tmp_work/pytorch.sif"
common = ("--num_steps=100000 --eval_num_batches=3 --eval_batch_size=128 "
          "--eval_num_samples=50 --eval_freq=100000 --save_freq=50000 --print_freq=20000")

runs = [
    ("ndp_d2", 2, "bridge=false,resample=5,timesteps=100"),
    ("ctx_d2", 2, "bridge=true,anchor=context,gamma_mode=linear,gamma_scale=1.0,anchor_loss_scale=1.0,resample=5,timesteps=100"),
    ("ndp_d3", 3, "bridge=false,resample=5,timesteps=100"),
    ("ctx_d3", 3, "bridge=true,anchor=context,gamma_mode=linear,gamma_scale=1.0,anchor_loss_scale=1.0,resample=5,timesteps=100"),
    ("ctx_d2b", 2, "bridge=true,anchor=context,gamma_mode=linear,gamma_scale=1.0,anchor_loss_scale=0.3,resample=5,timesteps=100"),
    ("ctx_d3b", 3, "bridge=true,anchor=context,gamma_mode=linear,gamma_scale=1.0,anchor_loss_scale=0.3,resample=5,timesteps=100"),
]

os.makedirs(base + "/jobs", exist_ok=True)
for expid, dim, margs in runs:
    script = "#!/bin/bash\n"
    script += "cd %s\n" % flowdir
    script += ("singularity exec --nv --bind /home/jianxu:/home/jianxu %s \\\n" % sif)
    script += ("  python gp_hd.py --model=nbpd --kernel=rbf --expname=hd "
               "--dim_x=%d --expid=%s \\\n" % (dim, expid))
    script += ("  --model_args %s \\\n" % margs)
    script += ("  %s\n" % common)
    path = base + "/jobs/job_%s.sh" % expid
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)
    print("wrote", path)
print("DONE")
