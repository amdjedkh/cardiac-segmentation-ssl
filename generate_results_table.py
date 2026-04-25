"""
generate_results_table.py
Generates a publication-quality results table as both PDF and PNG.

Usage:
    python generate_results_table.py --output_dir .

Outputs:
    results_table.pdf
    results_table.png
"""

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def generate_table(output_dir):
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    columns = ["Condition", "LVBP Dice", "LVMYO Dice", "RVBP Dice", "FG Dice", "FG IoU"]
    rows = [
        ["Scratch (random init)", "0.761", "0.355", "0.356", "0.491", "0.349"],
        ["Pretrained (SSL encoder)", "0.735", "0.358", "0.358", "0.484", "0.339"],
    ]

    # ------------------------------------------------------------------ #
    # Figure setup
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")

    # ------------------------------------------------------------------ #
    # Colors
    # ------------------------------------------------------------------ #
    header_color = "#2c3e50"
    row_colors = ["#f7f9fc", "#eef2f7"]
    text_color_header = "white"
    text_color_body = "#1a1a1a"
    best_color = "#1a6b3c"   # dark green for best values per column
    border_color = "#c8d0da"

    n_cols = len(columns)
    n_rows = len(rows)

    col_widths = [0.28] + [0.144] * (n_cols - 1)
    col_positions = []
    x = 0
    for w in col_widths:
        col_positions.append(x)
        x += w

    row_height = 0.28
    header_y = 0.72
    table_width = sum(col_widths)
    table_height = row_height * (n_rows + 1)

    # ------------------------------------------------------------------ #
    # Determine best value per numeric column (cols 1 onward)
    # ------------------------------------------------------------------ #
    best_per_col = {}
    for ci in range(1, n_cols):
        vals = [float(rows[ri][ci]) for ri in range(n_rows)]
        best_per_col[ci] = max(vals)

    # ------------------------------------------------------------------ #
    # Draw header
    # ------------------------------------------------------------------ #
    for ci, (col, xpos, cw) in enumerate(zip(columns, col_positions, col_widths)):
        rect = FancyBboxPatch(
            (xpos, header_y), cw, row_height,
            boxstyle="square,pad=0",
            linewidth=0.5, edgecolor=border_color,
            facecolor=header_color,
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(
            xpos + cw / 2, header_y + row_height / 2,
            col,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color=text_color_header,
            transform=ax.transAxes
        )

    # ------------------------------------------------------------------ #
    # Draw rows
    # ------------------------------------------------------------------ #
    for ri, row in enumerate(rows):
        y = header_y - (ri + 1) * row_height
        bg = row_colors[ri % 2]
        for ci, (val, xpos, cw) in enumerate(zip(row, col_positions, col_widths)):
            rect = FancyBboxPatch(
                (xpos, y), cw, row_height,
                boxstyle="square,pad=0",
                linewidth=0.5, edgecolor=border_color,
                facecolor=bg,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)

            # Bold + green for best value in column
            is_best = ci >= 1 and float(val) == best_per_col[ci]
            ax.text(
                xpos + cw / 2, y + row_height / 2,
                val,
                ha="center", va="center",
                fontsize=9.5,
                fontweight="bold" if is_best else "normal",
                color=best_color if is_best else text_color_body,
                transform=ax.transAxes
            )

    # ------------------------------------------------------------------ #
    # Title and caption
    # ------------------------------------------------------------------ #
    ax.text(
        table_width / 2, header_y + row_height + 0.06,
        "Segmentation Results on ACDC Test Set",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        color="#1a1a1a",
        transform=ax.transAxes
    )
    ax.text(
        table_width / 2, header_y - n_rows * row_height - 0.06,
        "Split: 70 train / 15 val / 15 test (seed=1). "
        "Both conditions use identical architecture and training protocol. "
        "Bold green values indicate best per column.",
        ha="center", va="top",
        fontsize=8, color="#555555",
        transform=ax.transAxes
    )

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    png_path = os.path.join(output_dir, "results_table.png")
    pdf_path = os.path.join(output_dir, "results_table.pdf")

    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_table(args.output_dir)


if __name__ == "__main__":
    main()
