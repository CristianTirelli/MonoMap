from pathlib import Path
from typing import Callable
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from analytics.models import AnalyticsRepresentativesColumn

# font Charter
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Charter"

def plot_mean_median_barcharts(
    BASE_PATH: Path,
    rows: list[AnalyticsRepresentativesColumn],

    get_values: Callable[[AnalyticsRepresentativesColumn, str], dict],
    data_type: str,

    algo_names: dict[str, str] | None = None,

    figsize: tuple[int, int] = (30, 16.5),
    BAR_WIDTH: int = 0.25,
    INNER_BAR_SPACING: int = 0.01,
    GROUP_SPACING: int = 1.5,

    MISSING_X_SIZE: int = 7.5,
    MISSING_X_THICKNESS: int = 2,

    # TOP_MARGIN_PIXELS: int = 400,

    LEGEND_FONT_SIZE = 40,

    GLOBAL_TITLE_FONT_SIZE = 30,
    GLOBAL_TITLE_FONT_WEIGHT = 500,

    GLOBAL_AXIS_LABEL_FONT_SIZE = 21,
    GLOBAL_AXIS_LABEL_FONT_WEIGHT = 400,
    GLOBAL_AXIS_LABEL_PAD = 18,

    GLOBAL_TICK_FONT_SIZE = 25,
    GLOBAL_TICK_LINE_WIDTH = 2,
    GLOBAL_TICK_LINE_LENGTH = 8,


    LOG_SCALE: bool = True,

    opacity_by_norm: bool = False,
    inverse_normalized_opacity: bool = False,

    id: str = "",

    debug: bool = False
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


    # normalization check
    def collect_values(attr: str) -> list[float]:
        collected: list[float] = []

        for row in rows:
            values_dict = get_values(row, attr)

            for v in values_dict.values():
                if v is not None:
                    collected.append(v)

        return collected

    mean_values = collect_values("mean")
    median_values = collect_values("median")

    EPS = 1e-6
    mean_is_normalized = all(0.0 - EPS <= v <= 1.0 + EPS for v in mean_values)
    median_is_normalized = all(0.0 - EPS <= v <= 1.0 + EPS for v in median_values)

    # mean and median must agree on scale
    assert mean_is_normalized == median_is_normalized, (
        f"{BASE_PATH} Mean and median scales differ: "
        f"is mean normalized: {mean_is_normalized}, "
        f"is median normalized: {median_is_normalized}"
    )

    is_normalized = mean_is_normalized


    # dynamic figsize on number of groups
    figsize = (max(figsize[0], n_groups), figsize[1])

    if n_rows <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_rows)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / n_rows) for i in range(n_rows)]

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

    if LOG_SCALE and not is_normalized:
        ax_mean.set_yscale("log")
        ax_median.set_yscale("log")

    def plot_chart(ax: plt.Axes, attr_name: str, title: str):
        max_value = 0.0

        for row in rows:
            values_dict = get_values(row, attr_name)

            for v in values_dict.values():
                if v is not None:
                    max_value = max(max_value, v)

        for row_idx, row in enumerate(rows):
            values_dict = get_values(row, attr_name)

            # AFTER
            timeout_dict = (
                row.algorithm_mean_timeout
                if attr_name == "mean"
                else row.algorithm_median_timeout
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

                timed_out = timeout_dict.get(key, True)
                if value is None:
                    values.append(np.nan)
                    missing_positions.append(offsets[key_idx])
                else:
                    if timed_out:
                        values.append(np.nan)
                        missing_positions.append(offsets[key_idx])
                    else:
                        values.append(value)

            for key_idx, (offset, value) in enumerate(zip(offsets, values)):
                if not np.isnan(value):
                    alpha = (value if not inverse_normalized_opacity else 1.0 - value) if is_normalized else 1.0
                    if alpha < 0.01:
                        alpha = 0.1

                    ax.bar(
                        offset,
                        value,
                        width=BAR_WIDTH,
                        color=colors[row_idx],
                        edgecolor="black",
                        alpha=alpha if opacity_by_norm else 1.0,
                    )

            if LOG_SCALE:
                cross_y = max_value * 1e-3
            else:
                cross_y = 0 if max_value > 0 else -0.1

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

        # ax.set_title(title, fontsize=20)
        # ax.tick_params(axis="both", labelsize=15)
        # ax.yaxis.label.set_size(15)
        ax.set_title(
            title,
            fontsize=GLOBAL_TITLE_FONT_SIZE,
            fontweight=GLOBAL_TITLE_FONT_WEIGHT,
            pad=GLOBAL_AXIS_LABEL_PAD
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=GLOBAL_TICK_FONT_SIZE,
            width=GLOBAL_TICK_LINE_WIDTH,
            length=GLOBAL_TICK_LINE_LENGTH
        )

        ax.grid(axis="y", linestyle="--", alpha=0.8)

    mean_title: str = "Algorithm Mean" + (f" {data_type}" if data_type != "" else "")
    median_title: str = "Algorithm Median" + (f" {data_type}" if data_type != "" else "")

    # draw
    plot_chart(ax_mean, "mean", mean_title)
    plot_chart(ax_median, "median", median_title)

    # x ticks
    ax_mean.set_xticks(group_centers)
    ax_median.set_xticks(group_centers)

    ax_mean.set_xticklabels(
        all_keys,
        rotation=45,
        ha="right",
        fontsize=GLOBAL_AXIS_LABEL_FONT_SIZE,
        fontweight=GLOBAL_AXIS_LABEL_FONT_WEIGHT)
    ax_median.set_xticklabels(
        all_keys,
        rotation=45,
        ha="right",
        fontsize=GLOBAL_AXIS_LABEL_FONT_SIZE,
        fontweight=GLOBAL_AXIS_LABEL_FONT_WEIGHT)


    # legend
    # bars
    legend_elements = [
        Patch(
            facecolor=colors[i],
            edgecolor="black",
            label=(
                lambda algo_type: next(
                    (v for k, v in (algo_names or {}).items() if k in algo_type),
                    f"{f'{row.macro_group}\n' if isinstance(row, AnalyticsRepresentativesColumn) else ''}"
                    f"{f'{row.id} ' if isinstance(row, AnalyticsRepresentativesColumn) else ''}"
                    f"{algo_type}"
                )
            )(row.sa_algorithm_type)
        )
        for i, row in enumerate(rows)
    ]

    # x
    legend_elements.append(
        Line2D(
            [0], [0],
            marker="x",
            color="red",
            label="Timed out",
            markersize=MISSING_X_SIZE * 1.5,
            markeredgewidth=MISSING_X_THICKNESS * 1.5,
            linestyle="None",
        )
    )

    COLUMNS = 4

    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.99),
        ncol=COLUMNS,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        handlelength=1.5,
        labelspacing=1.2,
        columnspacing=2.0,
        prop={"size": 22},
    )

    for txt in legend.get_texts():
        txt.set_multialignment("left")

    # top margin
    legend_rows = int(np.ceil(len(rows) / COLUMNS))
    top_margin = 0.9 - (legend_rows - 1) * 0.05
    top_margin = max(0.75, min(0.9, top_margin))
    fig.subplots_adjust(top=top_margin, hspace=0.55)

    # fig.tight_layout()

    norm_label: str = (("-inverse-opacity-norm" if inverse_normalized_opacity else "-opacity-norm") if opacity_by_norm else "-norm") if is_normalized else ""
    name: str = f"barchart-means-medians{f"-{data_type.lower()}" if data_type != "" else ""}{norm_label}{id}.png"
    location: Path = BASE_PATH / name
    print(f"Saving mean median barchart plot to {str(location)}")
    plt.savefig(
        location,
        dpi=150
    )
    plt.close()
