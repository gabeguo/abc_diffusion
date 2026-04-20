#!/bin/bash
#
#SBATCH --account=m1266
#SBATCH --job-name=eval_abc
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=16:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=32       # CPUs for the job
#SBATCH --ntasks=1            # Number of tasks (one per GPU)
#SBATCH --output=slurm_logs/%j.out

# Perlmutter:
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0/targets/x86_64-linux/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0/lib64:$LD_LIBRARY_PATH # TODO: may not be necessary, depending on your cluster
export HF_HOME=$SCRATCH/cache_sub # TODO: wherever your HF cache should be
dir_prefix="/pscratch/sd/g/gabeguo/abc_storage/DUMMY_RUNS" # TODO: change to whatever you want to store results in

# # Brev:
# export HF_HOME=/data/hf/cache_sub
# dir_prefix="/data"
<<alt_settings
DATASET_NAME="CelebV-HQ"
METHOD_NAMES=(
    "abc/volatility_0.5/with_bb"
    "abc/volatility_0.5/no_bb"
    "cond_bridge/volatility_0.5"
    "cond_bridge/volatility_0.125"
    "cond_bridge/volatility_0.09"
    "noise_to_data/cosine/alpha_3.0_eps_0.04"
    "noise_to_data/cosine/alpha_5.0_eps_0.05"
    "noise_to_data/exp/B_4.0_K_2.5"
    "noise_to_data/exp/B_5.0_K_5.0"
)
METHOD_NAME=${METHOD_NAMES[1]} # TODO: pick whatever method you want
CKPT="celebvhq/$METHOD_NAME/0240000.pt"
alt_settings

DATASET_NAME="Sky-timelapse"
METHOD_NAMES=(
    "abc/volatility_0.4/no_bb"
    "abc/volatility_0.4/with_bb"
    "cond_bridge/volatility_0.6"
    "cond_bridge/volatility_0.4"
    "cond_bridge/volatility_0.3"
    "cond_bridge/volatility_0.1"
    "cond_bridge/volatility_0.071"
    "noise_to_data/cosine/alpha_3.0_eps_0.04"
    "noise_to_data/cosine/alpha_5.0_eps_0.05"
    "noise_to_data/exp/B_4.0_K_2.5"
    "noise_to_data/exp/B_5.0_K_5.0"
)
METHOD_NAME=${METHOD_NAMES[0]} # TODO: pick whatever method you want
CKPT="sky_timelapse/$METHOD_NAME/0240000.pt"

FVD_VIDEOMAE_CKPT="/pscratch/sd/g/gabeguo/DiT/pretrained_models/vit_g_hybrid_pt_1200e_ssv2_ft.pth" # TODO: set this path to wherever it's stored

echo "Running inference for method: $METHOD_NAME"
echo "Using ckpt: $CKPT"
echo "Using dataset name: $DATASET_NAME"

curr_date=04-19-26

if [ "$DATASET_NAME" == "Sky-timelapse" ]; then
    latents_folder="${dir_prefix}/datasets/latents/sky_timelapse/res-256x256-fpc-64/test"
    subsample_everys=(2 4)
elif [ "$DATASET_NAME" == "CelebV-HQ" ]; then
    latents_folder="${dir_prefix}/datasets/latents/celebv_hq/res-256x256-fpc-32/train"
    subsample_everys=(1 2)
else
    echo "Invalid dataset name: $DATASET_NAME"
    exit 1
fi

inference_mode="NON_causal" # "causal"

if [ "$inference_mode" == "NON_causal" ]; then
    causal_flag="--teacher-force-pinned"
    pin_everys=(4 8 16)
    prefix_lengths=(1)
elif [ "$inference_mode" == "causal" ]; then
    causal_flag="--causal"
    pin_everys=(1)
    prefix_lengths=(1 4)
else
    echo "Invalid inference mode: $inference_mode"
    exit 1
fi

for subsample_every in ${subsample_everys[@]}; do
    for pin_every in ${pin_everys[@]}; do
    for prefix_length in ${prefix_lengths[@]}; do
        for num_sampling_steps in 250 500; do
            if [ "$inference_mode" == "NON_causal" ]; then
                prompt_setting_name="pin_every_${pin_every}"
            elif [ "$inference_mode" == "causal" ]; then
                prompt_setting_name="prefix_length_${prefix_length}"
            else
                echo "Invalid inference mode: $inference_mode"
                exit 1
            fi
            python sample_non_markov.py \
                $causal_flag --ckpt $CKPT \
                --batch-size 16 \
                --frames-decoded-per-call 2 \
                --max-num-batches 128 \
                --num-sampling-steps ${num_sampling_steps} \
                --seed 0 \
                --out_folder ${dir_prefix}/results_eval/${DATASET_NAME}/${curr_date}/${inference_mode}/subsample_${subsample_every}/${prompt_setting_name}/steps_${num_sampling_steps}/${METHOD_NAME} \
                --save-interval 32 \
                --nrow 4 \
                --subsample-every ${subsample_every} \
                --pin-every ${pin_every} \
                --prefix-length ${prefix_length} \
                --latents-folder ${latents_folder} \
                --fvd-model videomae \
                --fvd-videomae-ckpt $FVD_VIDEOMAE_CKPT \
                --ignore-last-frame \
                --clip-dt-eps 1e-6
        done
    done
    done
done