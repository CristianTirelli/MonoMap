import math
from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import networkx as nx
from networkx.classes.digraph import DiGraph

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, rc

# Could be improved and made into a class


def save_figure(
        fig: Figure,
        file_name: str,
        save_to_directory: str = None) -> str:
    file_path: str = file_name
    if save_to_directory:
        # make location path if not present
        location = Path(save_to_directory)
        location.mkdir(parents=True, exist_ok=True)
        file_path = str(location / file_path)

    fig.savefig(file_path)
    plt.close(fig)
    return file_path


def saveFigDFG(
        dfg: DiGraph,
        schedule: dict[str, list[int]],
        save_to_directory: str = None):
    """
    It plots the DFG following the schedule structure
    
    :param dfg: The DFG graph
    :type dfg: DiGraph

    :param schedule: The time schedule, that DFG nodes have to follow
    :type schedule: dict[str, list[int]]
    """
    red_edges = []
    black_edges = []

    for e in dfg.edges(data=True):
        if e[2].get("type") == 'back_dep':
            red_edges.append(e)
        else:
            black_edges.append(e)

    # pos defines positions of nodes as dict: index -> [pos_x, pos_y]
    # given schedule we can draw DFG in a orderly manner
    # we start by top left and add +0.5 pos_x for same time schedule and +0.5 pos_y for next time schedule
    step_x_new_clock = 0.05
    step_x = 0.5
    step_y = 2

    II = len(schedule)
    pos = {}
    for t in schedule:
        pos_y = step_y * (II - t)
        pos_x = step_x_new_clock * (t % 4)
        for n in schedule[t]:
            pos[n] = [pos_x, pos_y]
            pos_x += step_x

    fig, _ = plt.subplots(figsize=(30, 15))
    nx.draw_networkx_nodes(dfg, pos, node_size = 500)
    nx.draw_networkx_labels(dfg, pos)
    nx.draw_networkx_edges(dfg, pos, edgelist=red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(dfg, pos, edgelist=black_edges, arrows=True)

    save_figure(fig, 'dfg.png', save_to_directory)
    plt.close()


def save_plot_mappings_cgra(
        node_pe: dict[int, int],
        schedule: dict[str, list[int]],
        size_x: int,
        size_y: int,
        save_to_directory: str = None,
        id: None | str = None):
    II = len(schedule)

    # pos defines positions of nodes as dict: index -> [pos_x, pos_y]
    # given schedule and node PE we can draw CGRA in a orderly manner
    # for each II, we draw a size_x * size_y CGRA, and add each node_pe elements to its PE
    # we space to the right each CGRA
    CGRA = nx.Graph()

    size = size_x * size_y
    # build II CGRAs 
    for t in range(II):
        start_idx = t * size
        end_idx = start_idx + size

        for i in range(start_idx, end_idx):
            i_mod_size = (i % size)

            # top and bottom
            if i_mod_size < size - size_x:
                CGRA.add_edge(i, i + size_x)
            
            # right and left
            if (i_mod_size + 1) % size_x != 0:
                CGRA.add_edge(i, i + 1)


    step_x_new_clock_time = 0.3 if size_x < 10 and size_y < 10 else (150 if size_x < 20 and size_y < 20 else 8000)
    step_x = 0.25 if size_x < 10 else (75 if size_x < 20 else 2000)
    step_y = 0.25 if size_y < 10 else (75 if size_y < 20 else 2000)
    pos = {}
    for i in range(II):
        offset_clock_time = ((size_x - 1) * step_x + step_x_new_clock_time) * i

        start_idx = i * size
        for n in range(start_idx, start_idx + size):
            n_local = n % size

            col = n_local % size_x
            row = n_local // size_x

            pos[n] = [offset_clock_time + col * step_x, (size_y - 1 - row) * step_y]

    used_pe = []
    label_pe = []
    for t in schedule:
        for n in schedule[t]:
            used_pe.append(t * size + node_pe[n])
            label_pe.append(str(n))

    NODE_SIZE = 1000 if size_x < 10 and size_y < 10 else (400 if size_x < 20 and size_y < 20 else 150)
    FONT_SIZE = 20 if size_x < 10 and size_y < 10 else (12 if size_x < 20 and size_y < 20 else 7)

    plt_size = 6
    fig, ax = plt.subplots(figsize=(II * plt_size, plt_size))
    nx.draw_networkx_nodes(CGRA, pos, alpha=0.3, node_size=NODE_SIZE, ax=ax)

    nx.draw_networkx_nodes(CGRA, pos, nodelist=used_pe, node_color='red', node_size=NODE_SIZE, ax=ax)
    nx.draw_networkx_labels(CGRA, pos, labels=dict(zip(used_pe, label_pe)), font_size=FONT_SIZE, ax=ax)
    
    nx.draw_networkx_edges(CGRA, pos, arrows=False, ax=ax)

    return save_figure(fig, 'mappings-CGRA.png' if not id else f'mappings-CGRA-{id}.png', save_to_directory)


def plot_unique_figure_iteratons(
        x: int,
        temperatures: list[int],
        costs: list[int],
        xscale_log_cost: bool = True,
        yscale_log_cost: bool = True,
        xscale_log_temp: bool = True,
        yscale_log_temp: bool = True,
        ax1: Axes = None,
        **kwargs: dict) -> Figure:
    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=(30, 15))
    else:
        fig = ax1.get_figure()

    X = range(1, x + 1)

    # cost
    color = 'tab:blue'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('cost', color=color)

    ax1.plot(X, costs, color=color, label="Cost")

    costs_sma_slow: None | list[float] = kwargs.get("costs_sma_slow", None)
    if costs_sma_slow is not None and len(costs_sma_slow) > 0:
        ax1.plot(X, costs_sma_slow, color='violet', label="SMA Slow", alpha=0.8)

    costs_sma_fast: None | list[float] = kwargs.get("costs_sma_fast", None)
    if costs_sma_fast is not None and len(costs_sma_fast) > 0:
        ax1.plot(X, costs_sma_fast, color='yellow', label="SMA Fast", alpha=0.4)

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x')

    if xscale_log_cost:
        ax1.set_xscale('log')
    if yscale_log_cost:
        ax1.set_yscale('log')

    # temperature
    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('temperature', color=color)

    tround = [round(float(t), 2) if isinstance(t, (float, np.float64)) else t for t in temperatures]
    ax2.plot(X, tround, color=color, label="Temperature", alpha=0.4)
    ax2.tick_params(axis='y', labelcolor=color)

    if xscale_log_temp:
        ax2.set_xscale('log')
    if yscale_log_temp:
        ax2.set_yscale('log')

    ax1.set_title('Cost and Temperature')
    ax1.grid(True)

    # plot both legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    return fig


def save_plot_unique_figure_iteratons(
        x: int,
        temperatures: list[int],
        costs: list[int],
        xscale_log_cost: bool = True,
        yscale_log_cost: bool = True,
        xscale_log_temp: bool = True,
        yscale_log_temp: bool = False,
        ax1: Axes = None,
        save_to_directory: str = None,
        **kwargs: dict) -> Figure:
    fig =  plot_unique_figure_iteratons(
        x,
        temperatures,
        costs,
        xscale_log_cost,
        yscale_log_cost,
        xscale_log_temp,
        yscale_log_temp,
        ax1,
        **kwargs
    )
    return save_figure(fig, 'costs-and-temperatures.png', save_to_directory)


# we try to write a window aggregation plot ot reduce the amount of iterations printed as to
# produce better visual graph
# https://deepdatawithmivaa.com/what-is-windowed-aggregation/
def adaptive_block_reduce(x: int, *arrays: list[list[float]], max_points=250) -> tuple[int, ...]:
    """
    Reduce arrays to at most max_points by aggregating windows averages and preserves the last point
    """
    if arrays is None or len(arrays) == 0:
        print(arrays)
        raise AssertionError("At least temperature and costs should be passed in")

    if x <= max_points:
        return (x, *arrays)

    window = int(x / max_points)

    reduced_arrays = []
    for arr in arrays:
        if arr is None:
            reduced_arrays.append(None)
            continue

        arr = np.array(arr)
        reduced = []

        for i in range(0, x - window, window):
            reduced.append(arr[i:i + window].mean())

        last_chunk = arr[x - window:x - 1]
        if len(last_chunk) > 0:
            reduced.append(last_chunk.mean())

        reduced.append(arr[-1])
        reduced_arrays.append(np.array(reduced))

    return (len(reduced_arrays[0]), *reduced_arrays)


def save_plot_unique_figure_windows_reduced_iteratons(
    x: int,
    temperatures: list[float],
    costs: list[float],
    save_to_directory: str = None,
    **kwargs: dict):
    arrays_to_reduce = []
    keys = []
    for k, v in kwargs.items():
        if isinstance(v, list):
            arrays_to_reduce.append(v)
            keys.append(k)

    # windows
    new_x, costs_r, temps_r, *arrays_r = adaptive_block_reduce(
        x,
        costs,
        temperatures,
        *arrays_to_reduce
    )

    fig =  plot_unique_figure_iteratons(
        new_x,
        temps_r,
        costs_r,
        yscale_log_temp=False,
        **dict(zip(keys, arrays_r))
    )
    return save_figure(fig, 'windows-reduced-costs-and-temperatures.png', save_to_directory)


# We divide the iterations set into three sets and plot three differen graphs
def save_plot_three_figures_iterations(
    x: int,
    temperatures: list[float],
    costs: list[float],
    save_to_directory = None,
    **kwargs: dict):
    fig, axes = plt.subplots(3, 1, figsize=(30, 45))
    axes: list[Axes]

    index_step = x // 3
    for i in range(3):
        if i == 2:
            start, end = i * index_step, x - 1
        else:
            start, end = i * index_step, (i + 1) * index_step

        c_slice = costs[start:end]
        t_slice = temperatures[start:end]
        
        sliced_kwargs = {
            k: v[start:end] if isinstance(v, list) else v
            for k, v in kwargs.items()
        }

        plot_unique_figure_iteratons(
            end - start, 
            t_slice, 
            c_slice, 
            False,
            False,
            False,
            False,
            axes[i], 
            **sliced_kwargs
        )
        axes[i].set_title(f"Cost and Temperature: From iteration {start} to {end}")
    fig.tight_layout()
    return save_figure(fig, 'three-plots-costs-and-temperatures.png', save_to_directory)


# we take x amount of temperature runs, we normalize cost [100, 0] and we plot lines one over the other
# we then print the earlier run in a lighter color, and later run in darker color
def save_plot_overlay_cycles_hot_aligned(
    temperatures: list[float],
    costs: list[float],
    save_to_directory: str = None,
):
    temperatures: np.ndarray = np.array(temperatures)
    costs: np.ndarray = np.array(costs)

    # find temperature cycles: t[i - 1] < t[i]
    reset_indices = [0]
    for i, t in enumerate(temperatures):
        if 0 < i and temperatures[i - 1] < t:
            reset_indices.append(i)
    reset_indices.append(len(temperatures))

    n_cycles = len(reset_indices) - 1

    # find longest cycle length: used as baseline x
    cycle_lengths = [reset_indices[i + 1] - reset_indices[i] for i in range(n_cycles)]
    max_len = max(cycle_lengths)

    fig, ax = plt.subplots(figsize=(20, 10))
    cmap = cm.get_cmap("hot")

    for i in range(n_cycles):
        start = reset_indices[i]
        end = reset_indices[i + 1]

        # normalize costs from 100 to 0 linearly
        cost_chunk = costs[start:end]
        c_min = 0
        c_max = cost_chunk.max()
        diff = c_max - c_min
        cost_chunk_norm = 100 * ((cost_chunk - c_min) / diff) if diff > 0 else cost_chunk

        # from cycle index get color intensity
        intensity = i / max(1, n_cycles - 1)
        color = cmap(intensity)

        # plot
        ax.plot(
            range(len(cost_chunk_norm)),
            cost_chunk_norm,
            color=color,
            alpha=0.85,
            linewidth=0.8
        )

    ax.set_title("Overlay of Temperature Cycles (Hot Intensity)")
    ax.set_xlabel(f"Iterations from 0 to {max_len}")
    ax.set_ylabel("Normalized Cost [0, 100]")
    ax.grid(True)
    return save_figure(fig, 'overlay-cycles-costs-and-temperatures.png', save_to_directory)

# call this function to plot all graphs to directory
def save_plot_run_graphs(
    # CGRA required
    node_pe: dict[int, int],
    schedule: dict[str, list[int]],
    size_x: int,
    size_y: int,

    # temperature and cost required
    x: int,
    temperatures: list[float],
    probabilities: list[float],
    costs: list[float],

    # BENCHMARKS paths required
    save_to_directory: str = None,
    id: str = None,

    # extra: {"temp_cost": {}}
    **kwargs: dict):
    extra_temp_cost = kwargs.get("temp_cost")

    folder_path: str | None = save_to_directory
    if save_to_directory:
        if not id:
            raise AssertionError("There should be at least \"id\" defined for save_to_directory to work properly")
        folder_path = Path(save_to_directory) / id

    # save CGRA mappings
    save_plot_mappings_cgra(node_pe, schedule, size_x, size_y, folder_path)

    FONT_SIZE = 25
    rc('font', **{'size': FONT_SIZE})

    # save all plots
    save_plot_unique_figure_iteratons(x, probabilities, costs, save_to_directory=folder_path, **extra_temp_cost)
    save_plot_unique_figure_windows_reduced_iteratons(x, probabilities, costs, folder_path, **extra_temp_cost)
    save_plot_three_figures_iterations(x, probabilities, costs, folder_path, **extra_temp_cost)
    save_plot_overlay_cycles_hot_aligned(temperatures, costs, folder_path)

    # save freely on root
    return str(folder_path) if folder_path else folder_path










# TODO refine


def plot_malus_heatmaps(
    time_node_pe_malus: dict[str, dict[int, list[int]]],
    size_x: int,
    size_y: int
):
    """
    Plots a heatmap per node showing malus intensity over the size_x * size_y grid.
    """
    # flatten malus per node across all time steps
    node_malus: dict[int, list[int]] = {}
    for t in time_node_pe_malus:
        for n, malus in time_node_pe_malus[t].items():
            if n not in node_malus:
                node_malus[n] = [0] * (size_x * size_y)
            for pe, value in enumerate(malus):
                node_malus[n][pe] += value

    nodes = sorted(node_malus.keys())
    cols = 4
    rows = math.ceil(len(nodes) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes: Axes = axes.flatten()

    for i, n in enumerate(nodes):
        ax: Axes = axes[i]
        grid = np.array(node_malus[n]).reshape(size_x, size_y)
        im = ax.imshow(grid, cmap="hot", interpolation="nearest")
        ax.set_title(f"Node {n}")
        ax.set_xlabel("Col")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PE Malus Heatmap per Node")
    plt.tight_layout()
    plt.show()

def visit_heuristics_plot_run_graphs(
    size_x: int,
    size_y: int,
    
    time_node_pe_visits: dict[str, dict[int, list[int]]],
    time_node_pe_malus: dict[str, dict[int, list[int]]],
    
    plot_to_path: str | None
    ):
    plot_malus_heatmaps(time_node_pe_malus, size_x, size_y)
