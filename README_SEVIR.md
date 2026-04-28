# SEVIR VIL Benchmark

This repository includes a dedicated SEVIR VIL benchmark path for extreme-weather generation / nowcasting.

This benchmark is set up for four SEVIR runs:

- `ABC Non-Causal`: SKY-style training/evaluation with future pinned conditioning and sparse pinned rollout
- `ABC Causal`: causal training with past-only conditioning and causal rollout over a context prefix / forecast horizon split
- `Conditional Diffusion Bridge`: auxiliary-time bridge baseline
- `Noise-to-Data Diffusion`: auxiliary-time noise-to-data baseline
- all four runs operate on the same 25-frame SEVIR clips, save comparable metrics, and can be replotted from saved artifacts without rerunning evaluation

Implemented methods:

- `ABC`: the repo's physical-time non-Markov autoregressive diffusion bridge model.
- `Conditional Diffusion Bridge`: the repo's auxiliary-time conditional diffusion bridge variant.
- `Noise-to-Data Diffusion`: the repo's auxiliary-time noise-to-data variant.

Implemented components:

- Public-data downloader from the official SEVIR AWS Open Data bucket: [`download_sevir_vil.py`](download_sevir_vil.py)
- Raw SEVIR VIL dataset loader with train/val/test split logic: [`sevir_benchmark/data.py`](sevir_benchmark/data.py)
- Unified bridge-model training: [`train_sevir_abc.py`](train_sevir_abc.py)
- Shared evaluation with MAE / MSE / RMSE and thresholded CSI / POD / FAR / bias: [`eval_sevir_benchmark.py`](eval_sevir_benchmark.py)
- Replot-friendly figure renderer that consumes saved evaluation artifacts: [`create_sevir_figures.py`](create_sevir_figures.py)
- End-to-end shell wrapper: [`_run_sevir_benchmark.sh`](_run_sevir_benchmark.sh)

Default forecast setup:

- VIL only
- 25-frame clip
- `13` context frames and `12` forecast frames
- center clip start at frame `12`
- official train/test split around `2019-06-01`, with a validation fraction carved out of pre-cutoff events
- default model: `DiTXA-B/2`
- default SDE: uniform volatility (`A=0`, `K=1`)

Notes on comparability:

- this SEVIR path is a repo-specific benchmark rather than an attempt to mirror every preprocessing or split choice used elsewhere in the SEVIR literature
- evaluation must use the same `--method` that was used for training; overriding a checkpoint to a different method is intentionally rejected
- evaluation also locks `--prefix-length`, `--target-length`, and `--clip-starts` to the checkpoint's training configuration
- threshold and pixelwise metrics are computed at the native SEVIR spatial resolution; if the model was trained at a smaller `--image-size`, predictions are upsampled before metrics are computed
- for `conditional_diffusion_bridge` and `noise_to_data_diffusion`, `--num-sampling-steps` is applied per forecast bridge segment rather than once across the whole clip
- evaluation saves `plot_data.pt` bundles for selected batches, including native-resolution targets, ensemble predictions, conditioning frames, and metadata for figure regeneration
- `_run_sevir_benchmark.sh` runs exactly four configurations by default, writing outputs under:
  `results/sevir/{abc_noncausal,abc_causal,conditional_diffusion_bridge,noise_to_data_diffusion}`

Quick start:

```bash
python download_sevir_vil.py --output-root datasets/sevir --years 2017 2018 2019

python train_sevir_abc.py \
  --data-root datasets/sevir \
  --results-dir results/sevir/abc_noncausal \
  --method abc \
  --brownian-bridge-residual

python train_sevir_abc.py \
  --data-root datasets/sevir \
  --results-dir results/sevir/abc_causal \
  --method abc \
  --force-causal

python train_sevir_abc.py \
  --data-root datasets/sevir \
  --results-dir results/sevir/conditional_diffusion_bridge \
  --method conditional_diffusion_bridge

python train_sevir_abc.py \
  --data-root datasets/sevir \
  --results-dir results/sevir/noise_to_data_diffusion \
  --method noise_to_data_diffusion

python eval_sevir_benchmark.py \
  --ckpt results/sevir/abc_causal/checkpoints/epoch_0020.pt \
  --data-root datasets/sevir \
  --out-dir results/sevir/eval/abc_causal \
  --causal \
  --conditioning-prefix-length 13 \
  --pin-every 1

python create_sevir_figures.py \
  --eval-root results/sevir/eval \
  --methods abc_noncausal abc_causal conditional_diffusion_bridge noise_to_data_diffusion \
  --batch-name batch_0000 \
  --example-index 0 \
  --out-dir results/sevir/figures
```

Notes:

- The downloader uses anonymous `boto3` access to the official `s3://sevir` bucket, so AWS credentials are not required.
- The dataset loader reads raw HDF5 VIL files directly instead of duplicating the dataset into a second training format.
- Threshold metrics are configurable via `--thresholds`.
- The training script stores `--method` in each checkpoint, and the evaluator requires that same method at evaluation time.
- If `--num-ensemble-members > 1`, evaluation averages finalized per-member metrics instead of thresholding the ensemble mean field.
- `_run_sevir_benchmark.sh` trains `ABC Non-Causal`, `ABC Causal`, `Conditional Diffusion Bridge`, and `Noise-to-Data Diffusion`; running `_run_sevir_benchmark.sh --replot` skips download, training, and evaluation and rebuilds figures from saved `plot_data.pt` artifacts.
