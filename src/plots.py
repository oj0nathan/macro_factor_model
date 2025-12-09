# src/plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rolling_beta_heatmap(beta_df: pd.DataFrame,
                              factor_name: str = "REAL_FF"):
    """Return a matplotlib figure with a time–sector beta heatmap."""
    beta_plot = beta_df.dropna(how="all")

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        beta_plot.T.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    time_vals = beta_plot.index
    sec_vals = beta_plot.columns

    # X ticks
    if len(time_vals) > 0:
        num_xticks = min(8, len(time_vals))
        xtick_idx = np.linspace(0, len(time_vals) - 1, num_xticks, dtype=int)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels(
            [time_vals[i].strftime("%Y-%m") for i in xtick_idx],
            rotation=45,
        )

    ax.set_yticks(range(len(sec_vals)))
    ax.set_yticklabels(sec_vals)

    ax.set_title(f"Rolling sector sensitivity to {factor_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sector")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Beta to {factor_name}")

    plt.tight_layout()
    return fig
