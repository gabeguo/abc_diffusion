#!/bin/bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
if [ -z "${CONFIG_NAME}" ]; then
  echo "Usage: CONFIG_NAME=<abc_noncausal|abc_causal|conditional_diffusion_bridge|noise_to_data_diffusion> bash $0" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-datasets/sevir}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sevir}"
EVAL_ROOT="${EVAL_ROOT:-${RESULTS_ROOT}/eval}"

EPOCHS="${EPOCHS:-20}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
NUM_ENSEMBLE_MEMBERS="${NUM_ENSEMBLE_MEMBERS:-1}"
SAVE_GRID_EVERY="${SAVE_GRID_EVERY:-100}"
SAVE_PLOT_DATA_EVERY="${SAVE_PLOT_DATA_EVERY:-100}"
PLOT_MAX_ITEMS="${PLOT_MAX_ITEMS:-4}"
CLIP_DT_EPS="${CLIP_DT_EPS:-1e-6}"
SPLIT="${SPLIT:-test}"
MAX_EVENTS="${MAX_EVENTS:-}"
DISABLE_PY_ASSERTS="${DISABLE_PY_ASSERTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PREFIX_LENGTH="${PREFIX_LENGTH:-13}"
ABC_CAUSAL_PREFIX_LENGTH="${ABC_CAUSAL_PREFIX_LENGTH:-${PREFIX_LENGTH}}"
ABC_NONCAUSAL_PREFIX_LENGTH="${ABC_NONCAUSAL_PREFIX_LENGTH:-1}"
ABC_CAUSAL_PIN_EVERY="${ABC_CAUSAL_PIN_EVERY:-1}"
ABC_NONCAUSAL_PIN_EVERY="${ABC_NONCAUSAL_PIN_EVERY:-8}"
BASELINE_PREFIX_LENGTH="${BASELINE_PREFIX_LENGTH:-1}"
BASELINE_PIN_EVERY="${BASELINE_PIN_EVERY:-8}"

run_python() {
  local num_procs="$1"
  shift
  if [ "${num_procs}" -gt 1 ]; then
    if [ "${DISABLE_PY_ASSERTS}" = "1" ]; then
      PYTHONOPTIMIZE=1 "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${num_procs}" "$@"
    else
      "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${num_procs}" "$@"
    fi
  else
    if [ "${DISABLE_PY_ASSERTS}" = "1" ]; then
      PYTHONOPTIMIZE=1 "${PYTHON_BIN}" "$@"
    else
      "${PYTHON_BIN}" "$@"
    fi
  fi
}

case "${CONFIG_NAME}" in
  abc_noncausal)
    METHOD="abc"
    EVAL_FLAGS=(--conditioning-prefix-length "${ABC_NONCAUSAL_PREFIX_LENGTH}" --pin-every "${ABC_NONCAUSAL_PIN_EVERY}")
    ;;
  abc_causal)
    METHOD="abc"
    EVAL_FLAGS=(--causal --conditioning-prefix-length "${ABC_CAUSAL_PREFIX_LENGTH}" --pin-every "${ABC_CAUSAL_PIN_EVERY}")
    ;;
  conditional_diffusion_bridge)
    METHOD="conditional_diffusion_bridge"
    EVAL_FLAGS=(--conditioning-prefix-length "${BASELINE_PREFIX_LENGTH}" --pin-every "${BASELINE_PIN_EVERY}")
    ;;
  noise_to_data_diffusion)
    METHOD="noise_to_data_diffusion"
    EVAL_FLAGS=(--conditioning-prefix-length "${BASELINE_PREFIX_LENGTH}" --pin-every "${BASELINE_PIN_EVERY}")
    ;;
  *)
    echo "Unknown CONFIG_NAME: ${CONFIG_NAME}" >&2
    exit 1
    ;;
esac

CKPT_PATH="${RESULTS_ROOT}/${CONFIG_NAME}/checkpoints/epoch_$(printf '%04d' "${EPOCHS}").pt"
OUT_DIR="${EVAL_ROOT}/${CONFIG_NAME}"
mkdir -p "${OUT_DIR}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${MAX_EVENTS}" ]; then
  EXTRA_ARGS+=(--max-events "${MAX_EVENTS}")
fi
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "METHOD=${METHOD}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "EVAL_NUM_GPUS=${EVAL_NUM_GPUS}"
echo "BATCH_SIZE(per-rank)=${BATCH_SIZE}"
echo "NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS}"
echo "NUM_ENSEMBLE_MEMBERS=${NUM_ENSEMBLE_MEMBERS}"
echo "SPLIT=${SPLIT}"
if [ -n "${MAX_EVENTS}" ]; then
  echo "MAX_EVENTS=${MAX_EVENTS}"
fi
if [ "${DISABLE_PY_ASSERTS}" = "1" ]; then
  echo "PYTHONOPTIMIZE=1 (asserts disabled)"
fi

run_python "${EVAL_NUM_GPUS}" eval_sevir_benchmark.py \
  --method "${METHOD}" \
  --ckpt "${CKPT_PATH}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
  --num-ensemble-members "${NUM_ENSEMBLE_MEMBERS}" \
  --save-grid-every "${SAVE_GRID_EVERY}" \
  --save-plot-data-every "${SAVE_PLOT_DATA_EVERY}" \
  --plot-max-items "${PLOT_MAX_ITEMS}" \
  --clip-dt-eps "${CLIP_DT_EPS}" \
  "${EVAL_FLAGS[@]}" \
  "${EXTRA_ARGS[@]}"
