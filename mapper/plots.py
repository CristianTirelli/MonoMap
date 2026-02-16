
import networkx as nx
from networkx.classes.digraph import DiGraph

import matplotlib.pyplot as plt

def saveFigMappingsCGRA(node_pe: dict[int, int], schedule: dict[str, list[int]], size_x: int, size_y: int):
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
    plt.figure(figsize=(II * plt_size, plt_size))
    nx.draw_networkx_nodes(CGRA, pos, alpha=0.3, node_size=NODE_SIZE)

    nx.draw_networkx_nodes(CGRA, pos, nodelist=used_pe, node_color='red', node_size=NODE_SIZE)
    nx.draw_networkx_labels(CGRA, pos, labels=dict(zip(used_pe, label_pe)), font_size=FONT_SIZE)
    
    nx.draw_networkx_edges(CGRA, pos, arrows=False)
    plt.savefig('mappings-CGRA.png')
    plt.close()


def saveFigDFG(dfg: DiGraph, schedule: dict[str, list[int]]):
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

    plt.figure(figsize=(30, 15))
    nx.draw_networkx_nodes(dfg, pos, node_size = 500)
    nx.draw_networkx_labels(dfg, pos)
    nx.draw_networkx_edges(dfg, pos, edgelist=red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(dfg, pos, edgelist=black_edges, arrows=True)
    plt.savefig('dfg.png')
    plt.close()


def saveFigTemperature(temperatures: list[int], costs: list[int]):
    plt.figure(figsize=[50, 15])

    plt.plot(temperatures, costs, marker='o', linestyle='-', color='b')
    # plt probably sorts temperatures
    plt.gca().invert_xaxis()

    FONT_SIZE = 25

    plt.xlabel('Temperature', fontsize=FONT_SIZE)
    plt.ylabel('Cost', fontsize=FONT_SIZE)
    plt.title('Cost vs Temperature', fontsize=FONT_SIZE)
    plt.grid(True)

    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    plt.savefig("costs-vs-temperatures.png")
    plt.close()