#!/bin/bash
set -euo pipefail

REPLOT=0
for arg in "$@"; do
  case "$arg" in
    --replot)
      REPLOT=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

DATA_ROOT="${DATA_ROOT:-datasets/sevir}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sevir}"
EVAL_ROOT="${EVAL_ROOT:-${RESULTS_ROOT}/eval}"
FIGURE_ROOT="${FIGURE_ROOT:-${RESULTS_ROOT}/figures}"

FIGURE_BATCH_NAME="${FIGURE_BATCH_NAME:-batch_0000}"
FIGURE_EXAMPLE_INDEX="${FIGURE_EXAMPLE_INDEX:-0}"
FIGURE_FRAME_STRIDE="${FIGURE_FRAME_STRIDE:-2}"
FIGURE_MAX_FRAMES="${FIGURE_MAX_FRAMES:-8}"
FIGURE_NAME="${FIGURE_NAME:-sevir_qualitative}"

L_SUB="${L_SUB:-16}"
SAVE_GRID_EVERY="${SAVE_GRID_EVERY:-100}"
SAVE_PLOT_DATA_EVERY="${SAVE_PLOT_DATA_EVERY:-100}"
PLOT_MAX_ITEMS="${PLOT_MAX_ITEMS:-4}"
NUM_ENSEMBLE_MEMBERS="${NUM_ENSEMBLE_MEMBERS:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-${NUM_GPUS}}"

IMAGE_SIZE="${IMAGE_SIZE:-128}"
PREFIX_LENGTH="${PREFIX_LENGTH:-13}"
TARGET_LENGTH="${TARGET_LENGTH:-12}"
CLIP_STARTS="${CLIP_STARTS:-12}"
MODEL_NAME="${MODEL_NAME:-DiTXA-B/2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"

SDE_TYPE="${SDE_TYPE:-uniform_volatility}"
UNIFORM_SDE_A="${UNIFORM_SDE_A:-0.0}"
UNIFORM_SDE_K="${UNIFORM_SDE_K:-1.0}"

ABC_CAUSAL_PREFIX_LENGTH="${ABC_CAUSAL_PREFIX_LENGTH:-${PREFIX_LENGTH}}"
ABC_NONCAUSAL_PREFIX_LENGTH="${ABC_NONCAUSAL_PREFIX_LENGTH:-1}"
ABC_CAUSAL_PIN_EVERY="${ABC_CAUSAL_PIN_EVERY:-1}"
ABC_NONCAUSAL_PIN_EVERY="${ABC_NONCAUSAL_PIN_EVERY:-8}"
BASELINE_PREFIX_LENGTH="${BASELINE_PREFIX_LENGTH:-1}"
BASELINE_PIN_EVERY="${BASELINE_PIN_EVERY:-8}"

read -r -a CLIP_STARTS_ARR <<< "${CLIP_STARTS}"

run_python() {
  local num_procs="$1"
  shift
  if [ "${num_procs}" -gt 1 ]; then
    python -m torch.distributed.run --standalone --nproc_per_node="${num_procs}" "$@"
  else
    python "$@"
  fi
}

run_train() {
  local config_name="$1"
  local method="$2"
  local results_dir="${RESULTS_ROOT}/${config_name}"

  local extra_train_flags=()
  if [ "${config_name}" = "abc_causal" ]; then
    extra_train_flags+=(--force-causal)
  fi
  if [ "${config_name}" = "abc_noncausal" ]; then
    extra_train_flags+=(--brownian-bridge-residual)
  fi

  run_python "${NUM_GPUS}" train_sevir_abc.py \
    --data-root "${DATA_ROOT}" \
    --results-dir "${results_dir}" \
    --method "${method}" \
    --image-size "${IMAGE_SIZE}" \
    --prefix-length "${PREFIX_LENGTH}" \
    --target-length "${TARGET_LENGTH}" \
    --clip-starts "${CLIP_STARTS_ARR[@]}" \
    --model "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --l-sub "${L_SUB}" \
    --sde-type "${SDE_TYPE}" \
    --uniform_sde_A "${UNIFORM_SDE_A}" \
    --uniform_sde_K "${UNIFORM_SDE_K}" \
    "${extra_train_flags[@]}"
}

run_eval() {
  local config_name="$1"
  local method="$2"
  local ckpt="${RESULTS_ROOT}/${config_name}/checkpoints/epoch_$(printf '%04d' "${EPOCHS}").pt"
  local out_dir="${EVAL_ROOT}/${config_name}"

  local eval_flags=()
  case "${config_name}" in
    abc_causal)
      eval_flags+=(--causal --conditioning-prefix-length "${ABC_CAUSAL_PREFIX_LENGTH}" --pin-every "${ABC_CAUSAL_PIN_EVERY}")
      ;;
    abc_noncausal)
      eval_flags+=(--conditioning-prefix-length "${ABC_NONCAUSAL_PREFIX_LENGTH}" --pin-every "${ABC_NONCAUSAL_PIN_EVERY}")
      ;;
    conditional_diffusion_bridge|noise_to_data_diffusion)
      eval_flags+=(--conditioning-prefix-length "${BASELINE_PREFIX_LENGTH}" --pin-every "${BASELINE_PIN_EVERY}")
      ;;
    *)
      echo "Unknown config: ${config_name}" >&2
      exit 1
      ;;
  esac

  run_python "${EVAL_NUM_GPUS}" eval_sevir_benchmark.py \
    --ckpt "${ckpt}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${out_dir}" \
    --batch-size "${BATCH_SIZE}" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
    --num-ensemble-members "${NUM_ENSEMBLE_MEMBERS}" \
    --save-grid-every "${SAVE_GRID_EVERY}" \
    --save-plot-data-every "${SAVE_PLOT_DATA_EVERY}" \
    --plot-max-items "${PLOT_MAX_ITEMS}" \
    "${eval_flags[@]}"
}

if [ "${REPLOT}" -eq 0 ]; then
  python download_sevir_vil.py \
    --output-root "${DATA_ROOT}" \
    --years 2017 2018 2019

  run_train "abc_noncausal" "abc"
  run_train "abc_causal" "abc"
  run_train "conditional_diffusion_bridge" "conditional_diffusion_bridge"
  run_train "noise_to_data_diffusion" "noise_to_data_diffusion"

  run_eval "abc_noncausal" "abc"
  run_eval "abc_causal" "abc"
  run_eval "conditional_diffusion_bridge" "conditional_diffusion_bridge"
  run_eval "noise_to_data_diffusion" "noise_to_data_diffusion"
fi

python create_sevir_figures.py \
  --eval-root "${EVAL_ROOT}" \
  --methods abc_noncausal abc_causal conditional_diffusion_bridge noise_to_data_diffusion \
  --batch-name "${FIGURE_BATCH_NAME}" \
  --example-index "${FIGURE_EXAMPLE_INDEX}" \
  --frame-stride "${FIGURE_FRAME_STRIDE}" \
  --max-frames "${FIGURE_MAX_FRAMES}" \
  --out-dir "${FIGURE_ROOT}" \
  --output-name "${FIGURE_NAME}"
