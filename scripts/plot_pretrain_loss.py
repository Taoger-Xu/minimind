#!/usr/bin/env python3
"""Parse MiniMind pretraining logs and plot raw/smoothed loss curves."""

import argparse
import csv
import re
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([0-9.eE+-]+), "
    r"logits_loss: ([0-9.eE+-]+), aux_loss: ([0-9.eE+-]+), lr: ([0-9.eE+-]+)"
)


def parse_logs(paths):
    parsed = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            epoch, epochs, step, steps = map(int, match.group(1, 2, 3, 4))
            parsed.append({
                "epoch": epoch,
                "epochs": epochs,
                "local_step": step,
                "steps_per_epoch": steps,
                "loss": float(match.group(5)),
                "logits_loss": float(match.group(6)),
                "aux_loss": float(match.group(7)),
                "lr": float(match.group(8)),
                "source": path.name,
            })

    if not parsed:
        return []

    # A DDP run has fewer local steps per epoch than a single-GPU run. Normalize
    # every record to the largest (single-GPU-equivalent) epoch length so logs
    # from different world sizes share one monotonic x-axis. Later files/lines
    # replace replayed data after checkpoint recovery.
    canonical_steps_per_epoch = max(row["steps_per_epoch"] for row in parsed)
    records = {}
    for row in parsed:
        epoch_offset = (row["epoch"] - 1) * canonical_steps_per_epoch
        step_offset = round(
            row["local_step"] * canonical_steps_per_epoch / row["steps_per_epoch"]
        )
        global_step = epoch_offset + step_offset
        records[global_step] = {
            **row,
            "global_step": global_step,
            "canonical_steps_per_epoch": canonical_steps_per_epoch,
        }
    return [records[key] for key in sorted(records)]


def rolling_mean(values, window):
    queue, total, result = deque(), 0.0, []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        result.append(total / len(queue))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()

    records = parse_logs(args.logs)
    if not records:
        raise SystemExit("No pretraining loss records found")

    steps = [row["global_step"] for row in records]
    losses = [row["loss"] for row in records]
    smoothed = rolling_mean(losses, args.window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(steps, losses, color="#77aadd", alpha=0.32, linewidth=0.8, label="Raw loss")
    ax.plot(
        steps,
        smoothed,
        color="#cc3311",
        linewidth=2.2,
        label=f"Moving average ({args.window} log points)",
    )
    ax.scatter([steps[-1]], [losses[-1]], color="#222222", s=30, zorder=3)
    ax.annotate(
        f"equiv. step {steps[-1]:,}\nloss {losses[-1]:.4f}",
        (steps[-1], losses[-1]),
        xytext=(-85, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    ax.set_title("MiniMind Pretraining Loss")
    ax.set_xlabel("Single-GPU-equivalent training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*records[0].keys(), "smoothed_loss"])
            writer.writeheader()
            for row, smooth in zip(records, smoothed):
                writer.writerow({**row, "smoothed_loss": smooth})

    print(f"Parsed {len(records)} unique points: step {steps[0]} to {steps[-1]}")
    print(f"Saved plot: {args.output}")
    if args.csv:
        print(f"Saved data: {args.csv}")


if __name__ == "__main__":
    main()
