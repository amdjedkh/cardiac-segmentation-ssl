"""
plot_curves.py — Plot segmentation learning curves from CSV logs.

Usage:
    python plot_curves.py \
        --scratch  lightning_logs/version_16/metrics.csv \
        --pretrained lightning_logs/version_18/metrics.csv \
        --output learning_curves.png

Add more --pretrained paths as the run progresses and rerun.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def load_metrics(paths):
    if isinstance(paths, str):
        paths = [paths]
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df["val_Dice_FG"].notna()].reset_index(drop=True)
        dfs.append(df)
    combined = pd.concat(dfs).sort_values("epoch").reset_index(drop=True)
    return combined


def plot(scratch_paths, pretrained_paths, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot scratch runs
    for i, paths_group in enumerate([scratch_paths]):
        df = load_metrics(paths_group)
        label = "Scratch"
        ax.plot(df["epoch"], df["val_Dice_FG"],
                color="#3266ad", linewidth=2,
                marker="o", markersize=4,
                label=label)
        # Annotate final value
        ax.annotate(f"{df['val_Dice_FG'].iloc[-1]:.3f}",
                    xy=(df["epoch"].iloc[-1], df["val_Dice_FG"].iloc[-1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color="#3266ad")

    # Plot pretrained runs
    colors = ["#d85a30", "#1d9e75", "#7f77dd"]
    for i, paths_group in enumerate([pretrained_paths]):
        df = load_metrics(paths_group)
        label = "Pretrained (SSL)"
        color = colors[i % len(colors)]
        ax.plot(df["epoch"], df["val_Dice_FG"],
                color=color, linewidth=2, linestyle="--",
                marker="^", markersize=4,
                label=label)
        ax.annotate(f"{df['val_Dice_FG'].iloc[-1]:.3f}",
                    xy=(df["epoch"].iloc[-1], df["val_Dice_FG"].iloc[-1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=color)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val Dice FG", fontsize=12)
    ax.set_title("Segmentation finetuning: pretrained vs. scratch\n(ACDC, SA only, ED+ES frames)", fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(left=0)
    ax.axvspan(100, 184, alpha=0.08, color='gray')
    ax.text(142, 0.35, 'training\ninterrupted\n(compute\nbudget)', 
        ha='center', va='center', fontsize=7.5, color='#666666',
        style='italic')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch",     nargs="+", required=True,
                        help="Path(s) to scratch metrics.csv")
    parser.add_argument("--pretrained",  nargs="+", default=[],
                        help="Path(s) to pretrained metrics.csv")
    parser.add_argument("--output",      default="learning_curves.png")
    args = parser.parse_args()

    for p in args.scratch + args.pretrained:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    plot(args.scratch, args.pretrained, args.output)


if __name__ == "__main__":
    main()
