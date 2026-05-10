from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from analytics.models import AnalyticsRepresentativesColumn


def plot_mean_median_barcharts(
    BASE_PATH: Path,
    rows: list[AnalyticsRepresentativesColumn],

    figsize: tuple[int, int] = (30, 15),
    BAR_WIDTH: int = 0.25,
    INNER_BAR_SPACING: int = 0.01,
    GROUP_SPACING: int = 1.5,

    MISSING_X_SIZE: int = 6,
    MISSING_X_THICKNESS: int = 2,

    # TOP_MARGIN_PIXELS: int = 400,

    LOG_SCALE: bool = True,
):
    if not rows:
        raise ValueError("rows cannot be empty")
    
    # keys
    all_keys: list[str] = []
    for row in rows:
        for k in row.algorithm_mean_time.keys():
            if k not in all_keys:
                all_keys.append(k)
        for k in row.algorithm_median_time.keys():
            if k not in all_keys:
                all_keys.append(k)
    all_keys.sort()

    n_groups = len(all_keys)
    n_rows = len(rows)

    # dynamic figsize on number of groups
    figsize = (max(figsize[0], n_groups * 1.8), figsize[1])

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_rows)]

    # compute spacings
    total_group_width = (
        n_rows * BAR_WIDTH
        + (n_rows - 1) * INNER_BAR_SPACING
    )

    x = np.arange(n_groups) * (total_group_width + GROUP_SPACING)

    group_centers = x

    fig, (ax_mean, ax_median) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=False
    )
    ax_mean: plt.Axes
    ax_median: plt.Axes

    if LOG_SCALE:
        ax_mean.set_yscale("log")
        ax_median.set_yscale("log")

    def plot_chart(ax: plt.Axes, attr_name: str, title: str):

        max_value = 0.0

        for row in rows:
            values_dict = (
                row.algorithm_mean_time
                if attr_name == "mean"
                else row.algorithm_median_time
            )
            for v in values_dict.values():
                if v is not None:
                    max_value = max(max_value, v)

        for row_idx, row in enumerate(rows):

            values_dict = (
                row.algorithm_mean_time
                if attr_name == "mean"
                else row.algorithm_median_time
            )

            offsets = (
                group_centers
                - total_group_width / 2
                + row_idx * (BAR_WIDTH + INNER_BAR_SPACING)
                + BAR_WIDTH / 2
            )

            values = []
            missing_positions = []

            for key_idx, key in enumerate(all_keys):
                value = values_dict.get(key, None)

                if value is None:
                    values.append(np.nan)
                    missing_positions.append(offsets[key_idx])
                else:
                    values.append(value)

            ax.bar(
                offsets,
                values,
                width=BAR_WIDTH,
                color=colors[row_idx],
                edgecolor="black"
            )

            if LOG_SCALE:
                cross_y = max_value * 1e-3
            else:
                cross_y = -max_value * 0.03 if max_value > 0 else -0.1

            for mx in missing_positions:
                ax.plot(
                    mx,
                    cross_y,
                    marker="x",
                    color="red",
                    markersize=MISSING_X_SIZE,
                    markeredgewidth=MISSING_X_THICKNESS,
                    clip_on=False
                )

        if LOG_SCALE:
            lower = max(max_value * 1e-3, 1e-6)
        else:
            lower = -max_value * 0.08 if max_value > 0 else -0.2
        upper = max_value * 1.1 if max_value > 0 else 1

        ax.set_ylim(lower, upper)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        
    # draw
    plot_chart(ax_mean, "mean", "Algorithm Mean Time")
    plot_chart(ax_median, "median", "Algorithm Median Time")

    # x ticks
    ax_mean.set_xticks(group_centers)
    ax_median.set_xticks(group_centers)

    ax_mean.set_xticklabels(all_keys, rotation=45, ha="right")
    ax_median.set_xticklabels(all_keys, rotation=45, ha="right")

    # legend
    legend_elements = [
        Patch(
            facecolor=colors[i],
            edgecolor="black",
            label=f"{f"{row.macro_group}\n" if isinstance(row, AnalyticsRepresentativesColumn) else ""}{f"{row.id} " if isinstance(row, AnalyticsRepresentativesColumn)  else ""}{row.sa_algorithm_type}"
        )
        for i, row in enumerate(rows)
    ]

    COLUMNS = 3

    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=COLUMNS,
        frameon=True,
        borderaxespad=0.0,
        handlelength=1.5,
        labelspacing=1.2,
        columnspacing=2.0,
    )

    for txt in legend.get_texts():
        txt.set_multialignment("left")

    # top margin
    legend_rows = int(np.ceil(len(rows) / COLUMNS))
    top_margin = 0.9 - (legend_rows - 1) * 0.05
    top_margin = max(0.75, min(0.9, top_margin))
    fig.subplots_adjust(top=top_margin)

    location: Path = BASE_PATH / "barchart-means-medians.png"
    print(f"Saving mean median barchart plot to {str(location)}")
    plt.savefig(
        location,
        dpi=150
    )
