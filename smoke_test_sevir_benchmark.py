import os
import tempfile
from argparse import Namespace

import h5py
import pandas as pd
import torch

from models import DiT_models
from non_markov_diffusion.loss import dsm_loss, sample_p_base_x_t_cond_x_t_prev_x_t_next
from sevir_benchmark.abc import get_method_flags, rollout_bridge_model, sample_training_tuple, validate_benchmark_config
from sevir_benchmark.data import SevirVILDataset, _event_hash_bucket, inverse_normalize_vil
from sevir_benchmark.metrics import finalize_metric_state, init_metric_state, update_metric_state
from non_markov_diffusion.sde import UniformVolatilitySDE
from sevir_benchmark.visualization import render_rollout_figure, save_plot_data


class ZeroScoreNet(torch.nn.Module):
    def forward(self, x, t, t_next, y, cond_images, cond_times, cond_masks):
        return torch.zeros_like(x)


def choose_split_event_ids(seed: int, val_fraction: float):
    train_event_id = None
    val_event_id = None
    idx = 0
    while train_event_id is None or val_event_id is None:
        event_id = f"evt_pre_{idx:04d}"
        bucket = _event_hash_bucket(event_id, seed)
        if bucket < val_fraction and val_event_id is None:
            val_event_id = event_id
        if bucket >= val_fraction and train_event_id is None:
            train_event_id = event_id
        idx += 1
    return train_event_id, val_event_id


def make_tiny_sevir_root(root_dir: str):
    os.makedirs(os.path.join(root_dir, "data", "vil", "2019"), exist_ok=True)
    seed = 0
    val_fraction = 0.5
    train_event_id, val_event_id = choose_split_event_ids(seed=seed, val_fraction=val_fraction)
    catalog = pd.DataFrame(
        [
            {
                "event_id": train_event_id,
                "img_type": "vil",
                "file_name": "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5",
                "file_index": 0,
                "time_utc": "2019-01-01T00:00:00Z",
            },
            {
                "event_id": val_event_id,
                "img_type": "vil",
                "file_name": "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5",
                "file_index": 1,
                "time_utc": "2019-02-01T00:00:00Z",
            },
            {
                "event_id": "evt_test",
                "img_type": "vil",
                "file_name": "SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5",
                "file_index": 0,
                "time_utc": "2019-07-01T00:00:00Z",
            },
        ]
    )
    catalog.to_csv(os.path.join(root_dir, "CATALOG.csv"), index=False)

    first = torch.linspace(0, 255, 49 * 32 * 32).reshape(49, 32, 32).numpy().astype("float32")
    second = torch.flip(torch.from_numpy(first.copy()), dims=[0]).numpy().astype("float32")
    third = torch.roll(torch.from_numpy(first.copy()), shifts=5, dims=0).numpy().astype("float32")
    with h5py.File(os.path.join(root_dir, "data", "vil", "2019", "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"), "w") as f:
        f.create_dataset("vil", data=torch.stack([torch.from_numpy(first), torch.from_numpy(second)]).numpy())
    with h5py.File(os.path.join(root_dir, "data", "vil", "2019", "SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"), "w") as f:
        f.create_dataset("vil", data=third[None, ...])


def main():
    with tempfile.TemporaryDirectory() as root_dir:
        make_tiny_sevir_root(root_dir)

        train_ds = SevirVILDataset(root_dir=root_dir, split="train", image_size=32, prefix_length=13, target_length=12, val_fraction=0.5, seed=0)
        val_ds = SevirVILDataset(root_dir=root_dir, split="val", image_size=32, prefix_length=13, target_length=12, val_fraction=0.5, seed=0)
        test_ds = SevirVILDataset(root_dir=root_dir, split="test", image_size=32, prefix_length=13, target_length=12)
        test_native_ds = SevirVILDataset(
            root_dir=root_dir,
            split="test",
            image_size=32,
            prefix_length=13,
            target_length=12,
            return_native=True,
            return_metadata=True,
        )
        assert len(train_ds) == 1
        assert len(val_ds) == 1
        assert len(test_ds) == 1

        videos, labels = next(iter(torch.utils.data.DataLoader(test_ds, batch_size=1)))
        assert videos.shape == (1, 25, 1, 32, 32)
        assert labels.shape == (1,)
        resized_videos, native_labels, native_videos, metadata = next(iter(torch.utils.data.DataLoader(test_native_ds, batch_size=1)))
        assert resized_videos.shape == (1, 25, 1, 32, 32)
        assert native_videos.shape == (1, 25, 1, 32, 32)
        assert native_labels.shape == (1,)
        assert metadata["event_id"][0] == "evt_test"

        state = init_metric_state([16.0, 74.0])
        raw = inverse_normalize_vil(videos[:, 13:], normalize_mode=test_ds.normalize_mode)
        update_metric_state(state, pred_raw=raw, target_raw=raw)
        metrics = finalize_metric_state(state)
        assert abs(metrics["mae"]) < 1e-8
        assert metrics["csi_16"] > 0.999

        dummy_model = ZeroScoreNet()
        sde = UniformVolatilitySDE(A=0.0, K=1.0, score_network=dummy_model)
        y = torch.zeros(videos.shape[0], dtype=torch.long)
        validate_benchmark_config(
            args=Namespace(brownian_bridge_residual=True, l_sub=8),
            method="abc",
        )
        try:
            validate_benchmark_config(
                args=Namespace(brownian_bridge_residual=True, l_sub=8),
                method="conditional_diffusion_bridge",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("brownian_bridge_residual should remain ABC-only.")
        actual_model = DiT_models["DiTXA-S/8"](
            input_size=32,
            in_channels=1,
            num_classes=1,
            class_dropout_prob=0.0,
        )
        actual_model.eval()
        actual_sde = UniformVolatilitySDE(A=0.0, K=1.0, score_network=actual_model)

        for method in ("abc", "conditional_diffusion_bridge", "noise_to_data_diffusion"):
            flags = get_method_flags(method)
            batch = sample_training_tuple(
                videos=videos,
                prefix_length=13,
                method=method,
                device=videos.device,
                l_sub=8,
                force_causal=False,
            )
            frame_times = torch.linspace(0, 1, videos.shape[1], device=videos.device)
            nearest_gap = torch.abs(batch["t"].unsqueeze(1) - frame_times.unsqueeze(0)).min(dim=1).values
            assert torch.all(nearest_gap > 0.0)
            assert torch.all(batch["t_prev"] <= batch["t"])
            assert torch.all(batch["t_next"] > batch["t"])
            assert torch.all(batch["t_next"] <= 1.0)
            assert torch.all(nearest_gap >= 7.5e-4)
            assert torch.all(batch["cond_masks"][:, 0])
            model_t = batch["aux_tau"] if flags["aux_tau"] else batch["t"]
            bridge_t_prev = torch.zeros_like(batch["t_prev"]) if flags["aux_tau"] else batch["t_prev"]
            bridge_t_next = torch.ones_like(batch["t_next"]) if flags["aux_tau"] else batch["t_next"]
            x_t = sample_p_base_x_t_cond_x_t_prev_x_t_next(
                sde=sde,
                x_t_prev=torch.randn_like(batch["x_t_next"]) if flags["noise_to_data_diffusion"] else batch["x_t_prev"],
                x_t_next=batch["x_t_next"],
                t=model_t,
                t_prev=bridge_t_prev,
                t_next=bridge_t_next,
            )
            loss = dsm_loss(
                model=dummy_model,
                sde=sde,
                x_t=x_t,
                x_t_next=batch["x_t_next"],
                x_t_history=batch["cond_images"],
                t=model_t,
                t_next=batch["t_next"],
                t_history=batch["cond_times"],
                cond_masks=batch["cond_masks"],
                y=y,
                t_is_physical=(not flags["aux_tau"]),
            )
            assert torch.isfinite(loss)

            with torch.no_grad():
                actual_loss = dsm_loss(
                    model=actual_model,
                    sde=actual_sde,
                    x_t=x_t,
                    x_t_next=batch["x_t_next"],
                    x_t_history=batch["cond_images"],
                    t=model_t,
                    t_next=batch["t_next"],
                    t_history=batch["cond_times"],
                    cond_masks=batch["cond_masks"],
                    y=y,
                    t_is_physical=(not flags["aux_tau"]),
                )
            assert torch.isfinite(actual_loss)

            rollout = rollout_bridge_model(
                model=dummy_model,
                sde=sde,
                videos=videos,
                num_sampling_steps=32,
                method=method,
                causal=False,
                prefix_length=1,
                pin_every=8,
                clip_dt_eps=1e-6,
            )
            assert rollout["pred_frames"].shape == (1, 25, 1, 32, 32)
            assert rollout["eval_frame_mask"].shape == (25,)
            assert not rollout["eval_frame_mask"][0]
            assert not rollout["eval_frame_mask"][-1]

            actual_rollout = rollout_bridge_model(
                model=actual_model,
                sde=actual_sde,
                videos=videos,
                num_sampling_steps=32,
                method=method,
                causal=True,
                prefix_length=13,
                pin_every=1,
                clip_dt_eps=1e-6,
            )
            assert actual_rollout["pred_frames"].shape == (1, 25, 1, 32, 32)
            assert actual_rollout["num_solver_steps"] <= 32

        with tempfile.TemporaryDirectory() as output_dir:
            batch_dir = os.path.join(output_dir, "batch_0000")
            raw_videos = inverse_normalize_vil(native_videos)
            save_plot_data(
                output_dir=batch_dir,
                method="abc",
                batch_idx=0,
                pred_raw_mean=raw_videos,
                pred_raw_members=raw_videos.unsqueeze(0),
                target_raw=raw_videos,
                conditioning_raw=raw_videos,
                eval_frame_mask=torch.zeros(25, dtype=torch.bool),
                frame_times=torch.linspace(0, 1, 25),
                metadata={"event_id": ["evt_test"], "timestamp": ["2019-07-01T00:00:00+00:00"], "causal": False},
            )
            artifact_path = os.path.join(batch_dir, "plot_data.pt")
            assert os.path.isfile(artifact_path)
            render_rollout_figure(
                artifacts_by_label={
                    "ABC": torch.load(artifact_path, map_location="cpu", weights_only=False),
                    "Conditional Diffusion Bridge": torch.load(artifact_path, map_location="cpu", weights_only=False),
                },
                output_path=os.path.join(output_dir, "figure.png"),
                example_index=0,
                frame_indices=[0, 4, 8, 12],
                show_uncertainty=True,
                dpi=100,
            )
            assert os.path.isfile(os.path.join(output_dir, "figure.png"))

    print("SEVIR benchmark smoke test passed.")


if __name__ == "__main__":
    main()
