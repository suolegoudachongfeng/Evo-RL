#!/usr/bin/env python3
"""Watch resumed E299 action-expert-only run and stop it if it stops improving."""

from __future__ import annotations

import re
import statistics
import subprocess
import time
from pathlib import Path


SESSION = "resume_action_expert_E299_to10k"
LOG = Path("/mnt/project_eai/chp/checkpoints/resume_2mL_right_E299_prompt_v2_from_E249_evorl_action_expert_only_20260522_2gpu_5k_to10k.log")
START_STEP = 5000
TARGET_STEP = 10000
STEP_RE = re.compile(r"step:([0-9]+)(K?)\s+.*?loss:([0-9.]+)")


def parse_rows() -> list[tuple[int, float]]:
    if not LOG.exists():
        return []
    rows: list[tuple[int, float]] = []
    for line in LOG.read_text(errors="ignore").splitlines():
        if "ot_train.py:484 step:" not in line:
            continue
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1)) * (1000 if match.group(2) == "K" else 1)
        rows.append((step, float(match.group(3))))
    return rows


def running() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main() -> None:
    monitor_log = Path("/mnt/project_eai/chp/checkpoints/monitor_resume_action_expert_e299_early_stop_20260522.log")
    while True:
        rows = parse_rows()
        is_running = running()
        line = f"===== monitor tick {time.strftime('%F %T')} running={is_running} "
        if not rows:
            line += "no loss rows yet"
        else:
            step, loss = rows[-1]
            progressed = step - START_STEP
            if len(rows) >= 20:
                baseline = statistics.mean(v for _, v in rows[:10])
                recent = statistics.mean(v for _, v in rows[-10:])
                best_recent = min(v for _, v in rows[-20:])
                line += f"step={step} loss={loss:.5f} baseline={baseline:.5f} recent={recent:.5f}"
                stop = False
                if step < TARGET_STEP:
                    if progressed >= 2000 and recent > baseline * 1.01:
                        stop = True
                    if progressed >= 4000 and recent >= baseline * 0.995 and best_recent >= baseline * 0.99:
                        stop = True
                if stop and is_running:
                    line += f" EARLY_STOP killing {SESSION}"
                    subprocess.run(["tmux", "kill-session", "-t", SESSION], check=False)
            else:
                line += f"step={step} loss={loss:.5f} not enough points"
        with monitor_log.open("a") as f:
            f.write(line + "\n")
        if not is_running:
            with monitor_log.open("a") as f:
                f.write("No active monitored session remains. Exiting.\n")
            return
        time.sleep(300)


if __name__ == "__main__":
    main()

