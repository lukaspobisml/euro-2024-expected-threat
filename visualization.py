"""
visualization.py
Unified visualization module for Expected Threat (xT) analysis.
Ensures consistent fonts, subtitles, data labels, and styling across all charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global visual constants
BG_COLOR = "#d0d0d0"
COLOR_PASS = "#8faed6"   # Soft blue
COLOR_CARRY = "#e5836d"  # Soft terracotta
BORDER_COLOR = "#333333"
GRID_COLOR = "#888888"


def _add_bar_labels(ax, totals, x_offset=0.012):
    """Helper function to add bold total values at the end of each stacked bar."""
    for i, val in enumerate(totals):
        ax.text(
            val + x_offset,
            i,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#222222",
            fontweight="bold",
            fontfamily="sans-serif"
        )


def plot_stacked_xt_p90(
    df_players: pd.DataFrame,
    title: str = "Custom Tournament xT per 90 (EURO 2024 - 16x12 Grid)",
    top_n: int = 20,
    output_filename: str = None
):
    """
    Renders a standalone stacked horizontal bar chart with unified styling.
    """
    top_total = (
        df_players.sort_values(by="total_xt_p90", ascending=True)
        .tail(top_n)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(11.5, 8.0), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos = np.arange(len(top_total))
    pass_vals = top_total["pass_xt_p90"].values
    carry_vals = top_total["carry_xt_p90"].values
    total_vals = top_total["total_xt_p90"].values

    # Stacked horizontal bars
    ax.barh(
        y_pos,
        pass_vals,
        color=COLOR_PASS,
        height=0.68,
        label="Passes",
        edgecolor="#ffffff",
        linewidth=0.4
    )
    ax.barh(
        y_pos,
        carry_vals,
        left=pass_vals,
        color=COLOR_CARRY,
        height=0.68,
        label="Carries",
        edgecolor="#ffffff",
        linewidth=0.4
    )

    # Consistent data value labels
    _add_bar_labels(ax, total_vals)

    # Axis configuration
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_total["player"], fontsize=10, fontfamily="sans-serif")
    ax.set_xlabel("xT per 90", fontsize=11.5, fontfamily="sans-serif", labelpad=8)
    ax.set_ylabel("Player", fontsize=11.5, fontfamily="sans-serif", labelpad=8)
    ax.set_xlim(0, max(0.85, max(total_vals) * 1.12))

    # Titles and subtitle
    ax.set_title(title, fontsize=15, fontfamily="serif", pad=28, fontweight="bold")
    fig.text(
        0.5, 0.945,
        "Total bar = Total xT/90 | Blue = Pass threat | Orange = Progressive carry threat",
        fontsize=9.5,
        fontfamily="sans-serif",
        color="#444444",
        ha="center",
        style="italic"
    )

    # Grid and borders
    ax.grid(axis="x", linestyle="--", alpha=0.45, color=GRID_COLOR)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(0.8)

    # Legend
    ax.legend(
        loc="lower right",
        facecolor="#ffffff",
        edgecolor="none",
        framealpha=0.92,
        fontsize=9.5
    )

    # Footer
    fig.text(
        0.92, 0.015,
        "EURO 2024 | Min. 150 mins | Data: StatsBomb",
        fontsize=8.5,
        fontfamily="sans-serif",
        color="#444444",
        ha="right"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_filename:
        plt.savefig(output_filename, dpi=300)
    plt.show()


def plot_model_comparison(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    primary_name: str = "A: Custom Tournament Model (EURO 2024 16x12)",
    secondary_name: str = "B: Industry Benchmark Model (McKay Johns 12x8)",
    top_n: int = 15,
    output_filename: str = None
):
    """
    Renders side-by-side comparison charts with matching limits, labels, and fonts.
    """
    p_top = df_primary.sort_values(by="total_xt_p90", ascending=True).tail(top_n).copy()
    s_top = df_secondary.sort_values(by="total_xt_p90", ascending=True).tail(top_n).copy()

    # Determine unified x-axis limit so both subplots share the exact same scale
    max_x = max(p_top["total_xt_p90"].max(), s_top["total_xt_p90"].max())
    shared_xlim = max(0.85, max_x * 1.14)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.8), facecolor=BG_COLOR)

    # Shared subtitle above both subplots
    fig.text(
        0.5, 0.945,
        "Comparison of Total xT/90 | Blue = Pass threat | Orange = Progressive carry threat",
        fontsize=10,
        fontfamily="sans-serif",
        color="#444444",
        ha="center",
        style="italic"
    )

    # Subplot 1: Custom Tournament Model
    ax1.set_facecolor(BG_COLOR)
    y1 = np.arange(len(p_top))
    p_pass = p_top["pass_xt_p90"].values
    p_carry = p_top["carry_xt_p90"].values
    p_totals = p_top["total_xt_p90"].values

    ax1.barh(y1, p_pass, color=COLOR_PASS, height=0.65, label="Passes", edgecolor="#ffffff", linewidth=0.4)
    ax1.barh(y1, p_carry, left=p_pass, color=COLOR_CARRY, height=0.65, label="Carries", edgecolor="#ffffff", linewidth=0.4)
    _add_bar_labels(ax1, p_totals)

    ax1.set_yticks(y1)
    ax1.set_yticklabels(p_top["player"], fontsize=9.5, fontfamily="sans-serif")
    ax1.set_xlabel("xT per 90", fontsize=11, fontfamily="sans-serif")
    ax1.set_ylabel("Player", fontsize=11, fontfamily="sans-serif")
    ax1.set_xlim(0, shared_xlim)
    ax1.set_title(primary_name, fontsize=13, fontfamily="serif", pad=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.45, color=GRID_COLOR)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", facecolor="#ffffff", edgecolor="none", fontsize=9)
    for s in ax1.spines.values():
        s.set_color(BORDER_COLOR)
        s.set_linewidth(0.8)

    # Subplot 2: Benchmark Model
    ax2.set_facecolor(BG_COLOR)
    y2 = np.arange(len(s_top))
    s_pass = s_top["pass_xt_p90"].values
    s_carry = s_top["carry_xt_p90"].values
    s_totals = s_top["total_xt_p90"].values

    ax2.barh(y2, s_pass, color=COLOR_PASS, height=0.65, label="Passes", edgecolor="#ffffff", linewidth=0.4)
    ax2.barh(y2, s_carry, left=s_pass, color=COLOR_CARRY, height=0.65, label="Carries", edgecolor="#ffffff", linewidth=0.4)
    _add_bar_labels(ax2, s_totals)

    ax2.set_yticks(y2)
    ax2.set_yticklabels(s_top["player"], fontsize=9.5, fontfamily="sans-serif")
    ax2.set_xlabel("xT per 90", fontsize=11, fontfamily="sans-serif")
    ax2.set_xlim(0, shared_xlim)
    ax2.set_title(secondary_name, fontsize=13, fontfamily="serif", pad=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.45, color=GRID_COLOR)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", facecolor="#ffffff", edgecolor="none", fontsize=9)
    for s in ax2.spines.values():
        s.set_color(BORDER_COLOR)
        s.set_linewidth(0.8)

    # Shared footer
    fig.text(
        0.95, 0.015,
        "EURO 2024 | Min. 150 mins | Data: StatsBomb",
        fontsize=8.5,
        fontfamily="sans-serif",
        color="#444444",
        ha="right"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    if output_filename:
        plt.savefig(output_filename, dpi=300)
    plt.show()