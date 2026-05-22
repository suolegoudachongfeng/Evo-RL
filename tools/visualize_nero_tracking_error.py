#!/usr/bin/env python
"""Visualize NERO command tracking error from a LeRobot dataset.

The NERO adapter stores:
- observation.state: 28 dims
  left joints 7, right joints 7, left ee pose 6, right ee pose 6, grippers 2
- action: 14 dims
  left delta ee pose 6, right delta ee pose 6, grippers 2

For each pair of consecutive frames in the same episode, this script compares:
    planned_next_ee = current_ee + commanded_delta_ee
    actual_next_ee  = next_observation_ee

It is intentionally offline-first: it does not add latency to policy deployment.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_LEFT_EE = slice(14, 20)
STATE_RIGHT_EE = slice(20, 26)
ACTION_LEFT_DELTA = slice(0, 6)
ACTION_RIGHT_DELTA = slice(6, 12)
POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


def _arm_slices(arm: str) -> tuple[slice, slice]:
    if arm == "left":
        return STATE_LEFT_EE, ACTION_LEFT_DELTA
    if arm == "right":
        return STATE_RIGHT_EE, ACTION_RIGHT_DELTA
    raise ValueError(f"Unsupported arm: {arm}")


def _as_vec(value: object, expected_dim: int, column: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Column {column!r} expected dim {expected_dim}, got {arr.shape[0]}")
    return arr


def _load_dataset_frames(dataset_root: Path) -> pd.DataFrame:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    required = {"observation.state", "action", "episode_index", "frame_index"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def compute_tracking_errors(
    df: pd.DataFrame,
    command_column: str = "action",
    arms: tuple[str, ...] = ("left", "right"),
    min_command_norm: float = 0.0,
) -> pd.DataFrame:
    if command_column not in df.columns:
        raise KeyError(f"Command column {command_column!r} not found in dataset")

    records: list[dict[str, float | int | str]] = []
    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        if len(ep_df) < 2:
            continue

        for i in range(len(ep_df) - 1):
            cur = ep_df.iloc[i]
            nxt = ep_df.iloc[i + 1]
            current_state = _as_vec(cur["observation.state"], 28, "observation.state")
            next_state = _as_vec(nxt["observation.state"], 28, "observation.state")
            command = _as_vec(cur[command_column], 14, command_column)

            for arm in arms:
                state_slice, action_slice = _arm_slices(arm)

                current_ee = current_state[state_slice]
                actual_next_ee = next_state[state_slice]
                commanded_delta = command[action_slice]
                actual_delta = actual_next_ee - current_ee
                error = actual_delta - commanded_delta

                command_norm_xyz = float(np.linalg.norm(commanded_delta[:3]))
                if command_norm_xyz < min_command_norm:
                    continue

                record: dict[str, float | int | str] = {
                    "episode_index": int(episode_index),
                    "frame_index": int(cur["frame_index"]),
                    "arm": arm,
                    "command_norm_xyz": command_norm_xyz,
                    "actual_norm_xyz": float(np.linalg.norm(actual_delta[:3])),
                    "error_norm_xyz": float(np.linalg.norm(error[:3])),
                    "error_norm_rot": float(np.linalg.norm(error[3:])),
                    "error_squared_all": float(np.sum(error**2)),
                }
                for j, axis in enumerate(POSE_AXES):
                    record[f"command_{axis}"] = float(commanded_delta[j])
                    record[f"actual_delta_{axis}"] = float(actual_delta[j])
                    record[f"error_{axis}"] = float(error[j])
                records.append(record)

    return pd.DataFrame.from_records(records)


def compute_lag_sweep(
    df: pd.DataFrame,
    command_column: str = "action",
    arms: tuple[str, ...] = ("left", "right"),
    min_command_norm: float = 0.0,
    max_lag: int = 10,
) -> pd.DataFrame:
    if command_column not in df.columns:
        raise KeyError(f"Command column {command_column!r} not found in dataset")

    rows: list[dict[str, float | int | str]] = []
    for lag in range(1, max_lag + 1):
        per_lag_records = []
        for episode_index, ep_df in df.groupby("episode_index", sort=True):
            ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
            if len(ep_df) <= lag:
                continue

            states = np.stack([_as_vec(v, 28, "observation.state") for v in ep_df["observation.state"]])
            commands = np.stack([_as_vec(v, 14, command_column) for v in ep_df[command_column]])
            for i in range(len(ep_df) - lag):
                for arm in arms:
                    state_slice, action_slice = _arm_slices(arm)
                    commanded_delta = commands[i, action_slice]
                    command_norm_xyz = float(np.linalg.norm(commanded_delta[:3]))
                    if command_norm_xyz < min_command_norm:
                        continue

                    actual_delta = states[i + lag, state_slice] - states[i, state_slice]
                    error = actual_delta - commanded_delta
                    per_lag_records.append(
                        {
                            "episode_index": int(episode_index),
                            "arm": arm,
                            "lag": lag,
                            "command_norm_xyz": command_norm_xyz,
                            "actual_norm_xyz": float(np.linalg.norm(actual_delta[:3])),
                            "error_norm_xyz": float(np.linalg.norm(error[:3])),
                            "error_norm_rot": float(np.linalg.norm(error[3:])),
                        }
                    )

        if not per_lag_records:
            continue
        lag_df = pd.DataFrame.from_records(per_lag_records)
        for arm, arm_df in lag_df.groupby("arm"):
            rows.append(
                {
                    "arm": arm,
                    "lag": lag,
                    "steps": int(len(arm_df)),
                    "command_norm_xyz_mean": float(arm_df["command_norm_xyz"].mean()),
                    "actual_norm_xyz_mean": float(arm_df["actual_norm_xyz"].mean()),
                    "error_norm_xyz_mean": float(arm_df["error_norm_xyz"].mean()),
                    "error_norm_xyz_median": float(arm_df["error_norm_xyz"].median()),
                    "error_norm_xyz_p95": float(np.percentile(arm_df["error_norm_xyz"], 95)),
                    "error_norm_rot_mean": float(arm_df["error_norm_rot"].mean()),
                }
            )

    return pd.DataFrame.from_records(rows)


def compute_cumulative_window_errors(
    df: pd.DataFrame,
    command_column: str = "action",
    arms: tuple[str, ...] = ("left", "right"),
    min_command_norm: float = 0.0,
    windows: tuple[int, ...] = (3, 5, 10, 15, 30),
) -> pd.DataFrame:
    if command_column not in df.columns:
        raise KeyError(f"Command column {command_column!r} not found in dataset")

    rows: list[dict[str, float | int | str]] = []
    for window in windows:
        per_window_records = []
        for episode_index, ep_df in df.groupby("episode_index", sort=True):
            ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
            if len(ep_df) <= window:
                continue

            states = np.stack([_as_vec(v, 28, "observation.state") for v in ep_df["observation.state"]])
            commands = np.stack([_as_vec(v, 14, command_column) for v in ep_df[command_column]])
            for i in range(len(ep_df) - window):
                for arm in arms:
                    state_slice, action_slice = _arm_slices(arm)
                    commanded_delta = commands[i : i + window, action_slice].sum(axis=0)
                    command_norm_xyz = float(np.linalg.norm(commanded_delta[:3]))
                    if command_norm_xyz < min_command_norm:
                        continue

                    actual_delta = states[i + window, state_slice] - states[i, state_slice]
                    error = actual_delta - commanded_delta
                    per_window_records.append(
                        {
                            "episode_index": int(episode_index),
                            "arm": arm,
                            "window": window,
                            "command_norm_xyz": command_norm_xyz,
                            "actual_norm_xyz": float(np.linalg.norm(actual_delta[:3])),
                            "error_norm_xyz": float(np.linalg.norm(error[:3])),
                            "error_norm_rot": float(np.linalg.norm(error[3:])),
                        }
                    )

        if not per_window_records:
            continue
        window_df = pd.DataFrame.from_records(per_window_records)
        for arm, arm_df in window_df.groupby("arm"):
            rows.append(
                {
                    "arm": arm,
                    "window": window,
                    "steps": int(len(arm_df)),
                    "command_norm_xyz_mean": float(arm_df["command_norm_xyz"].mean()),
                    "actual_norm_xyz_mean": float(arm_df["actual_norm_xyz"].mean()),
                    "error_norm_xyz_mean": float(arm_df["error_norm_xyz"].mean()),
                    "error_norm_xyz_median": float(arm_df["error_norm_xyz"].median()),
                    "error_norm_xyz_p95": float(np.percentile(arm_df["error_norm_xyz"], 95)),
                    "error_norm_rot_mean": float(arm_df["error_norm_rot"].mean()),
                }
            )

    return pd.DataFrame.from_records(rows)


def _plot_over_time(result: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for arm, arm_df in result.groupby("arm"):
        axes[0].plot(arm_df["error_norm_xyz"].to_numpy(), label=f"{arm} xyz")
        axes[1].plot(arm_df["command_norm_xyz"].to_numpy(), label=f"{arm} command xyz")
    axes[0].set_ylabel("XYZ tracking error norm")
    axes[0].set_title("NERO EE Tracking Error Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Recorded step")
    axes[1].set_ylabel("Command XYZ norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_error_over_time.png", dpi=160)
    plt.close(fig)


def _plot_command_vs_actual(result: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(-1)
    for axis_i, axis_name in enumerate(("x", "y", "z")):
        ax = axes[axis_i]
        for arm, arm_df in result.groupby("arm"):
            ax.scatter(
                arm_df[f"command_{axis_name}"],
                arm_df[f"actual_delta_{axis_name}"],
                s=4,
                alpha=0.25,
                label=arm,
            )
        lim_values = result[[f"command_{axis_name}", f"actual_delta_{axis_name}"]].to_numpy().reshape(-1)
        finite = lim_values[np.isfinite(lim_values)]
        if len(finite):
            lo, hi = np.percentile(finite, [1, 99])
            pad = max((hi - lo) * 0.1, 1e-4)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
        ax.set_title(f"Command vs Actual Delta {axis_name.upper()}")
        ax.set_xlabel("commanded delta")
        ax.set_ylabel("actual delta")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for axis_i, axis_name in enumerate(("x", "y", "z"), start=3):
        ax = axes[axis_i]
        for arm, arm_df in result.groupby("arm"):
            ax.hist(arm_df[f"error_{axis_name}"], bins=80, alpha=0.45, label=arm)
        ax.set_title(f"Signed Error {axis_name.upper()}")
        ax.set_xlabel("actual delta - command")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "command_vs_actual_xyz.png", dpi=160)
    plt.close(fig)


def _plot_lag_sweep(lag_result: pd.DataFrame, output_dir: Path) -> None:
    if lag_result.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for arm, arm_df in lag_result.groupby("arm"):
        arm_df = arm_df.sort_values("lag")
        axes[0].plot(arm_df["lag"], arm_df["error_norm_xyz_mean"], marker="o", label=f"{arm} mean")
        axes[0].plot(
            arm_df["lag"],
            arm_df["error_norm_xyz_p95"],
            marker=".",
            linestyle="--",
            label=f"{arm} p95",
        )
        axes[1].plot(arm_df["lag"], arm_df["actual_norm_xyz_mean"], marker="o", label=f"{arm} actual")
    axes[0].set_title("Lag Sweep: command[t] vs actual displacement[t+lag]")
    axes[0].set_ylabel("XYZ error norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Lag in recorded frames")
    axes[1].set_ylabel("Actual XYZ displacement norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lag_sweep_xyz.png", dpi=160)
    plt.close(fig)


def _plot_cumulative_windows(cumulative_result: pd.DataFrame, output_dir: Path) -> None:
    if cumulative_result.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for arm, arm_df in cumulative_result.groupby("arm"):
        arm_df = arm_df.sort_values("window")
        ax.plot(arm_df["window"], arm_df["error_norm_xyz_mean"], marker="o", label=f"{arm} mean")
        ax.plot(arm_df["window"], arm_df["error_norm_xyz_p95"], marker=".", linestyle="--", label=f"{arm} p95")
    ax.set_title("Cumulative Window Error")
    ax.set_xlabel("Window size in recorded frames")
    ax.set_ylabel("XYZ error norm of summed command vs actual displacement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_window_error_xyz.png", dpi=160)
    plt.close(fig)


def _write_summary(
    result: pd.DataFrame,
    output_dir: Path,
    lag_result: pd.DataFrame | None = None,
    cumulative_result: pd.DataFrame | None = None,
) -> None:
    lines = []
    lines.append("# NERO Tracking Error Summary")
    lines.append("")
    lines.append(f"Total compared steps: {len(result)}")
    lines.append("")

    for arm, arm_df in result.groupby("arm"):
        lines.append(f"## {arm}")
        lines.append("")
        for key in ("command_norm_xyz", "actual_norm_xyz", "error_norm_xyz", "error_norm_rot"):
            values = arm_df[key].to_numpy(dtype=np.float64)
            lines.append(
                f"- `{key}`: mean={values.mean():.6g}, median={np.median(values):.6g}, "
                f"p95={np.percentile(values, 95):.6g}, max={values.max():.6g}"
            )
        lines.append("")

    if lag_result is not None and not lag_result.empty:
        lines.append("## Lag Sweep")
        lines.append("")
        for arm, arm_df in lag_result.groupby("arm"):
            best = arm_df.loc[arm_df["error_norm_xyz_mean"].idxmin()]
            lines.append(
                f"- `{arm}` best mean XYZ lag: lag={int(best['lag'])}, "
                f"mean_error={best['error_norm_xyz_mean']:.6g}, "
                f"p95_error={best['error_norm_xyz_p95']:.6g}"
            )
        lines.append("")

    if cumulative_result is not None and not cumulative_result.empty:
        lines.append("## Cumulative Window")
        lines.append("")
        for arm, arm_df in cumulative_result.groupby("arm"):
            best = arm_df.loc[arm_df["error_norm_xyz_mean"].idxmin()]
            lines.append(
                f"- `{arm}` best mean XYZ window: window={int(best['window'])}, "
                f"mean_error={best['error_norm_xyz_mean']:.6g}, "
                f"p95_error={best['error_norm_xyz_p95']:.6g}"
            )
        lines.append("")

    output_dir.joinpath("summary.md").write_text("\n".join(lines))


def visualize(
    result: pd.DataFrame,
    output_dir: Path,
    lag_result: pd.DataFrame | None = None,
    cumulative_result: pd.DataFrame | None = None,
) -> None:
    if result.empty:
        raise ValueError("No valid tracking records were produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "tracking_errors.csv", index=False)
    with open(output_dir / "tracking_errors.pkl", "wb") as f:
        pickle.dump(result, f)
    if lag_result is not None and not lag_result.empty:
        lag_result.to_csv(output_dir / "lag_sweep.csv", index=False)
    if cumulative_result is not None and not cumulative_result.empty:
        cumulative_result.to_csv(output_dir / "cumulative_window_errors.csv", index=False)

    _write_summary(result, output_dir, lag_result=lag_result, cumulative_result=cumulative_result)
    _plot_over_time(result, output_dir)
    _plot_command_vs_actual(result, output_dir)
    if lag_result is not None:
        _plot_lag_sweep(lag_result, output_dir)
    if cumulative_result is not None:
        _plot_cumulative_windows(cumulative_result, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to a local LeRobot dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV, pickle, and plots.")
    parser.add_argument(
        "--command-column",
        default="action",
        help="Vector column to compare against actual execution. Use 'action' for sent command.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("left", "right"),
        default=["left", "right"],
        help="Which arm(s) to analyze.",
    )
    parser.add_argument(
        "--min-command-norm",
        type=float,
        default=0.0,
        help="Ignore frames whose commanded XYZ delta norm is below this threshold.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=10,
        help="Maximum frame lag for lag sweep. Set 0 to disable.",
    )
    parser.add_argument(
        "--cumulative-windows",
        nargs="*",
        type=int,
        default=[3, 5, 10, 15, 30],
        help="Window sizes, in recorded frames, for cumulative command-vs-actual error. Empty disables it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_dataset_frames(args.dataset_root)
    result = compute_tracking_errors(
        df,
        command_column=args.command_column,
        arms=tuple(args.arms),
        min_command_norm=args.min_command_norm,
    )
    lag_result = (
        compute_lag_sweep(
            df,
            command_column=args.command_column,
            arms=tuple(args.arms),
            min_command_norm=args.min_command_norm,
            max_lag=args.max_lag,
        )
        if args.max_lag > 0
        else None
    )
    cumulative_result = (
        compute_cumulative_window_errors(
            df,
            command_column=args.command_column,
            arms=tuple(args.arms),
            min_command_norm=args.min_command_norm,
            windows=tuple(args.cumulative_windows),
        )
        if args.cumulative_windows
        else None
    )
    visualize(result, args.output_dir, lag_result=lag_result, cumulative_result=cumulative_result)
    print(f"Wrote tracking analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
