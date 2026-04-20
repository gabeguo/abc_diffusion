#!/bin/bash
#
#SBATCH --account=m1266
#SBATCH --job-name=train_abc
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=36:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=16       # CPUs for the job
#SBATCH --ntasks=4            # Number of tasks (one per GPU)
#SBATCH --output=slurm_logs/%j.out

# this on perlmutter
export HF_HOME=$SCRATCH/cache_sub # TODO: wherever your HF cache should be
dir_prefix="/pscratch/sd/g/gabeguo/abc_storage/DUMMY_RUNS" # TODO: change this to whatever directory you want to store results + data in (make sure it's LARGE: we need 100GB+ for data)
export WANDB_CACHE_DIR=/pscratch/sd/g/gabeguo/wandb_cache # TODO: wherever your wandb cache should be

# # this on GCP
# export NCCL_NET=Socket
# dir_prefix=".."

dataset_name="Sky-timelapse" # "CelebV-HQ" "Checkerboard" 
curr_date="04-19-26"

max_grad_norm=5.0
lr=1e-4

if [ "$dataset_name" == "Sky-timelapse" ]; then
    num_classes=1
    periodic_sde_k=1
    periodic_sde_alpha=0
    periodic_sde_eps=0.4
    results_dir="$dir_prefix/results/$curr_date/sky-timelapse/volatility_0.4/abc/no_bb"
    image_size=32
    global_batch_size=256
    model="DiTXA-B/2"
    data_folder="$dir_prefix/datasets/latents/sky_timelapse/res-256x256-fpc-64"
    latents_folder="$data_folder/train"
    eval_latents_folder="$data_folder/test"
    eval_subsample_every=4
elif [ "$dataset_name" == "CelebV-HQ" ]; then
    num_classes=1
    periodic_sde_k=1
    periodic_sde_alpha=0
    periodic_sde_eps=0.5
    results_dir="$dir_prefix/results/$curr_date/celebvhq/volatility_0.5/abc/no_bb"
    image_size=32
    global_batch_size=256
    model="DiTXA-B/2"
    latents_folder="$dir_prefix/datasets/latents/celebv_hq/res-256x256-fpc-32/train"
    eval_latents_folder=$latents_folder
    extra_flags="--latents-percent-train 0.95"
    eval_subsample_every=2
elif [ "$dataset_name" == "Checkerboard" ]; then
    num_classes=3
    periodic_sde_k=31
    periodic_sde_alpha=1.0
    periodic_sde_eps=0.04
    results_dir="results/$curr_date/checkerboard-qk-norm"
    image_size=32
    global_batch_size=128
    model="DiTXA-B/2"
    latents_folder="none"
    eval_latents_folder="none"
    eval_subsample_every=2
fi

DIFFUSION_MODE="" # ABC (no BB)
# DIFFUSION_MODE="--brownian-bridge-residual" # ABC (with BB)
# DIFFUSION_MODE="--aux-tau" # Conditional Diffusion Bridge
# DIFFUSION_MODE="--aux-tau --noise-to-data-diffusion" # Noise-to-Data Diffusion

python -m torch.distributed.run --nproc_per_node=4 train.py \
    $DIFFUSION_MODE --model $model \
    --global-batch-size $global_batch_size \
    --lr $lr \
    --max-grad-norm $max_grad_norm \
    --epochs 10000 \
    --warmup-steps 10000 \
    --cosine-decay-steps 290000 \
    --adam-beta2 0.999 \
    --num-classes $num_classes \
    --ckpt-every 20000 \
    --log-every 200 \
    --num-workers 16 \
    --sde-type periodic_volatility \
    --periodic_sde_alpha $periodic_sde_alpha \
    --periodic_sde_k $periodic_sde_k \
    --periodic_sde_eps $periodic_sde_eps \
    --margin-eps 1.00e-4 \
    --adjust-eps 1.01e-4 \
    --results-dir $results_dir \
    --image-size $image_size \
    --dataset-name $dataset_name \
    --latents-folder $latents_folder \
    --L-sub 16 \
    --eval-batch-size 16 \
    --eval-max-num-batches 4 \
    --eval-frames-decoded-per-call 2 \
    --eval-num-sampling-steps 1000 \
    --eval-seed 0 \
    --eval-snapshot-interval 1001 \
    --eval-nrow 4 \
    --eval-subsample-every $eval_subsample_every \
    --eval-pin-every 8 \
    --eval-prefix-length 1 \
    --eval-save-interval 4 \
    --eval-latents-folder $eval_latents_folder \
    --eval-clip-dt-eps 1e-6 \
    $extra_flags

