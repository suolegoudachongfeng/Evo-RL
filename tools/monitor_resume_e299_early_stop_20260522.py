#!/usr/bin/env python3
"""Watch resumed E299 policy runs and stop runs that stop improving early."""

from __future__ import annotations

import dataclasses
import re
import statistics
import subprocess
import time
from pathlib import Path


@dataclasses.dataclass
class Job:
    name: str
    session: str
    log: Path
    start_step: int
    target_step: int
    min_delta_steps: int = 2000
    hard_delta_steps: int = 4000


JOBS = [
    Job(
        name="evorl_from_scratch_to30k",
        session="resume_evorl_E299_to30k",
        log=Path("/mnt/project_eai/chp/checkpoints/resume_2mL_right_E299_prompt_v2_evorl_from_scratch_20260522_4gpu_15k_to30k.log"),
        start_step=15000,
        target_step=30000,
    ),
    Job(
        name="full_ft_to10k",
        session="resume_fullft_E299_to10k",
        log=Path("/mnt/project_eai/chp/checkpoints/resume_2mL_right_E299_prompt_v2_from_E249_evorl_full_ft_20260522_2gpu_5k_to10k.log"),
        start_step=5000,
        target_step=10000,
    ),
    Job(
        name="lora_fresh_to10k",
        session="lora_E299_2gpu_10k_20260522",
        log=Path("/mnt/project_eai/chp/checkpoints/2mL_right_E299_prompt_v2_from_E249_evorl_lora_r16_20260522_2gpu_10k.log"),
        start_step=0,
        target_step=10000,
    ),
]

STEP_RE = re.compile(r"step:([0-9]+)(K?)\s+.*?loss:([0-9.]+)")


def parse_rows(log: Path) -> list[tuple[int, float]]:
    if not log.exists():
        return []
    rows: list[tuple[int, float]] = []
    for line in log.read_text(errors="ignore").splitlines():
        if "ot_train.py:484 step:" not in line:
            continue
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1)) * (1000 if match.group(2) == "K" else 1)
        rows.append((step, float(match.group(3))))
    return rows


def tmux_has_session(session: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def stop_session(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], check=False)


def should_stop(job: Job, rows: list[tuple[int, float]]) -> tuple[bool, str]:
    if len(rows) < 20:
        return False, "not enough loss points yet"
    current_step = rows[-1][0]
    if current_step >= job.target_step:
        return False, "target reached or nearly reached"
    progressed = current_step - job.start_step
    if progressed < job.min_delta_steps:
        return False, f"only progressed {progressed} steps"

    first = rows[: min(10, len(rows))]
    recent = rows[-min(10, len(rows)) :]
    baseline = statistics.mean(loss for _, loss in first)
    recent_mean = statistics.mean(loss for _, loss in recent)
    best_recent = min(loss for _, loss in rows[-min(20, len(rows)) :])

    # At 2k, only stop if it is clearly worse. At 4k, stop if there is no real improvement.
    if progressed >= job.min_delta_steps and recent_mean > baseline * 1.01:
        return True, f"recent mean {recent_mean:.5f} worsened vs baseline {baseline:.5f} after {progressed} steps"
    if progressed >= job.hard_delta_steps and recent_mean >= baseline * 0.995 and best_recent >= baseline * 0.99:
        return True, f"no meaningful improvement after {progressed} steps: baseline {baseline:.5f}, recent {recent_mean:.5f}, best_recent {best_recent:.5f}"
    return False, f"still improving/acceptable: baseline {baseline:.5f}, recent {recent_mean:.5f}, progressed {progressed}"


def main() -> None:
    monitor_log = Path("/mnt/project_eai/chp/checkpoints/monitor_resume_e299_early_stop_20260522.log")
    monitor_log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        active = False
        lines = [f"===== monitor tick {time.strftime('%F %T')} ====="]
        for job in JOBS:
            rows = parse_rows(job.log)
            running = tmux_has_session(job.session)
            active = active or running
            if not rows:
                lines.append(f"{job.name}: running={running}, no loss rows yet")
                continue
            stop, reason = should_stop(job, rows)
            lines.append(f"{job.name}: running={running}, step={rows[-1][0]}, loss={rows[-1][1]:.5f}, {reason}")
            if running and stop:
                lines.append(f"{job.name}: EARLY_STOP killing tmux session {job.session}")
                stop_session(job.session)
        with monitor_log.open("a") as f:
            f.write("\n".join(lines) + "\n")
        if not active:
            with monitor_log.open("a") as f:
                f.write("No active monitored sessions remain. Exiting.\n")
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
