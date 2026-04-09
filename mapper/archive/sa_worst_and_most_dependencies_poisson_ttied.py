
import math
import random
import time

from networkx import DiGraph, Graph, shortest_path_length

import numpy as np

from benchmark import Benchmark
from plots import saveFigTemperature


# ALGORITHM:    The overall algorithm is a classica sa, we loop until the temperature freezes or we
#               reach a valid solution. We initialize the first solution at random and compute its cost.
#               Then from that solution we compute a NEIGHBOUR solution and evaluate it, if it is better
#               we keep it and if it is worse we randomly select based on temperature and cost delta if we
#               keep it (hill climb) or not.

# TEMPERATURE:  The simulated annealing temperature is attached to the algorithm performance, we use a simple
#               moving average to keep track of both a 201 costs SMA and 5 costs SMA, as soon as we reach
#               freezing temperature we check if the 5 costs SMA is under the 201 costs SMA, if it is we
#               add temperature and continue the search, as SMAs indicate that we are downtrending with cost.

# COST:         The cost is computed by looping over all dependency edges and check if they are respected,
#               if they are not we add the distance squared to the cost to discourage further away dependencies.
#               While calculating the cost we keep track of distances of nodes whose edges are not connected
#               and we build a list that holds operations indexes based on their cost, and the number of the
#               dependecies that they have (both incoming and outgoing), we add a fixed arbitrary value of 20
#               for each dependency then we sort them from worst to best, worst being the most distant with more
#               dependencies and best being the closest one with least dependencies.
#               We could remove the cost of dependencies for well placed nodes.

# NEIGHBOUR:    The routine that computes the neighbour solution is as follows: We take the schedule and the
#               previous solution and a list of nodes based on distance cost and dependencies, which has been
#               computed inside COST. Given these information we draw a random node following a poisson distribution
#               from the worst nodes, the node that we get is the one that will be moved. We maintain the current
#               solution for all schedules that do not have the selected node and for the one that has we select
#               a random PE position from the one available minus the current PE of such node.


class IntegerCircularBuffer():
    # delete not needed for now

    def __init__(self, size: int):
        self.size = size
        self.elements = [0 for _ in range(self.size)]
        self.tail = self.head = 0
        self.sma_prev = 0
        self.k = 1 / self.size

    def add_int(self, i: int):

        # compute next head position
        idx = 0 if self.head + 1 == self.size else self.head + 1

        # move tail if at capacity
        if idx == self.tail:
            self.tail = 0 if self.tail + 1 == self.size else self.tail + 1

        # add el and update head
        self.elements[idx] = i
        self.head = idx

    def last(self) -> int:
        return self.elements[self.tail]
    
    def next(self, cost: int) -> int:
        self.sma_prev = self.sma_prev + self.k * (cost - self.last())

        self.add_int(cost)
        return self.sma_prev


def isConnected(pe1: int, pe2: int, size_y: int, size_x: int) -> bool:
    """
    Returns true if `pe1` is conntected to `pe2` for a CGRA of size `size_x * size_y`, otherwise it returns false.

    Assumes that PEs connections happear only between adjacent PEs, where edge PEs wrap around.

    Args:
        pe1 (int): First PE
        pe2 (int): Second PE
        size_y (int): Number of PEs rows
        size_x (int): Number of PEs columns

    Returns:
        bool: True if `pe1` is conntected to `pe2`, otherwise false
    """
    i1 = pe1 // size_y
    j1 = pe1 % size_y

    i2 = pe2 // size_y
    j2 = pe2 % size_y

    # same row
    if i1 == i2:
        if (pe1 == pe2 + 1) or (pe1 == pe2 - 1):
            return True
        if abs(pe1 - pe2) == size_y - 1:
            return True

    # same col
    if j1 == j2:
        if (pe1 == pe2 + size_y) or (pe1 == pe2 - size_y):
            return True
        if abs(i1 - i2) == size_x - 1:
            return True

    # center
    if pe1 == pe2:
        return True

    return False


def random_sol_generator(schedule: dict[str, list[int]], size_x: int, size_y: int) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    Random solution generator.

    It assumes that the schedule does not schedule more instructions than available PEs.

    For each schedule it positions instructions at random in available PEs.
    
    :param schedule: The schedule of instructions
    :type schedule: dict[str, list[int]]

    :param size_y: The row size of the CGRE
    :type size_y: int

    :param size_x: The column size of the CGRE
    :type size_x: int

    :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be both valid and invalid
    :rtype: tuple[dict[int, int], dict[int, list[int]]]
    """
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}

    size = size_x * size_y

    for t in schedule:
        available_pes = [i for i in range(size)]

        for node in schedule[t]:
            pe = random.choice(available_pes)
            
            if pe not in pe_nodes:
                pe_nodes[pe] = []
            pe_nodes[pe].append(node)

            if node not in node_pe:
                node_pe[node] = pe
            else:
                print(f"should not happend, node: {node}, pe: {pe}")

            available_pes.remove(pe)
    return (node_pe, pe_nodes)


def neighbour_sol_generator(schedule: dict[str, list[int]], from_node_pe: dict[int, int], sol_worst_nodes: list[int], size_x: int, size_y: int) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    Neighbour solution generator: We keep from solution `from_pe_nodes` nodes `keep_nodes` at the PE they are bound to.

    It assumes that the schedule does not schedule more instructions than available PEs.

    For each schedule it positions instructions at random in available PEs.
    
    :param schedule: The schedule of instructions
    :type schedule: dict[str, list[int]]

    :param from_node_pe: The node -> pe mapping tobe used with `keep_nodes`
    :type from_node_pe: dict[int, int]

    :param keep_nodes: The list of nodes to keep fixed to PEs
    :type keep_nodes: list[int]

    :param size_y: The row size of the CGRE
    :type size_y: int

    :param size_x: The column size of the CGRE
    :type size_x: int

    :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be both valid and invalid
    :rtype: tuple[dict[int, int], dict[int, list[int]]]
    """
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}

    size = size_x * size_y

    # from poisson distribution get node
    nnodes = len(sol_worst_nodes)
    # lmda: int = nnodes - nnodes * 1
    node_to_move_idx = np.clip(np.random.poisson(1), 0, nnodes - 1)
    node_to_move = sol_worst_nodes[node_to_move_idx]

    for t in schedule:
        if node_to_move in schedule[t]:
            # move at random
            for n in schedule[t]:
                if n != node_to_move:
                    # maintain rest in schedule
                    pe = from_node_pe[n]
                    node_pe[n] = pe

                    if pe not in pe_nodes:
                        pe_nodes[pe] = []
                    pe_nodes[pe].append(n)

            # place it on remaining spots
            block_list = [node_pe[n] for n in schedule[t] if n != node_to_move]
            block_list.append(from_node_pe[node_to_move])
            allow_list = [i for i in range(size) if i not in block_list]

            new_pe = random.choice(allow_list)

            node_pe[node_to_move] = new_pe

            if new_pe not in pe_nodes:
                pe_nodes[new_pe] = []
            pe_nodes[new_pe].append(node_to_move)
        else:
            # maintain rest
            for n in schedule[t]:
                pe = from_node_pe[n]
                node_pe[n] = pe

                if pe not in pe_nodes:
                    pe_nodes[pe] = []
                pe_nodes[pe].append(n)

    return (node_pe, pe_nodes)


def pe_distance(arch: Graph, pe1: int, pe2: int) -> int:
    """
    Computes the distance from `pe1` to `pe2` in number of hops (edges) from source to target, respectively

    It assumes pe1 and pe2 are are already sized to the architecture, i.e. [0, size)

    :param arch: The architecture graph (MRRG)
    :type arch: Graph

    :param pe1: The source PE
    :type pe1: int

    :param pe2: The target PE
    :type pe2: int

    :return: The distance from pe1 to pe2 in number of edges
    :rtype: int
    """
    return shortest_path_length(arch, source=pe1, target=pe2)


def cost_space_solution(node_pe: dict[int, int], dfg: DiGraph, arch: Graph, size_x: int, size_y: int) -> tuple[int, list[int]]:
    """
    We need to add a value, a "cost" to all actions that randomness produces and that bring us away from a valid solution.

    We need to structure those costs in such a way that the more wrong the action is the more it costs.

    Wrong actions are, in order of more bad to less bad:

    Schedule two instructions on the same PE at the same scheduling time (Impossible)
    Do not respect dependencies, changes based on how far the two dependent instructions are

    So the lower the cost the better solution it is (near to a valid solution)

    We assume solutions that reach a cost of 0 (valid solution) are all equally good.
    
    -> We could distinguish them by "use the same PE as few times as possible" as a mean to reduce chip PE usage consumption
    
    :param pe_nodes: The solution schedule, a map PE -> instructions
    :type pe_nodes: dict[int, list[int]]

    :param dfg: The DFG graph dependencies
    :type dfg: DiGraph

    :return: The cost of the current solution. A cost of zero indicates that the solution is valid
    :rtype: int
    """
    cost = 0

    worst_nodes: dict[int, int] = {}

    for e in dfg.edges:
        source = e[0]
        destination = e[1]

        ps = node_pe[source]
        pd = node_pe[destination]

        d = 0
        # if not connected means that pd, ps are int that they are not adjacent
        if not isConnected(pd, ps, size_y, size_x):
            d = pe_distance(arch, pd, ps)
            # print(f"Source operation {source} at pe {pd} to operation {destination} at pe {ps} has distance {d}")
            cost += d ** 2

            # dfg is actually undirected Graph as in map we strip away directions
            worst_nodes[source] = d + dfg.degree(source) * 20
            worst_nodes[destination] = d + dfg.degree(destination) * 20
        else:
            if source not in worst_nodes:
                worst_nodes[source] = 0
            if destination not in worst_nodes:
                worst_nodes[destination] = 0

    return cost, [k for k, _ in sorted(worst_nodes.items(), key=lambda item: item[1], reverse=True)]


def simulatedAnnealingSearch(schedule: dict[str, list[int]], dfg: Graph, arch: Graph, II: int, size_x: int, size_y: int, BENCHMARK: Benchmark) -> tuple[dict[int, int],  dict[int, list[int]], float]:
    print("\n\n\n")
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}
    sol_cost: int
    sol_worst_nodes: list[int] = []

    # Data to study the search
    costs = []
    temperatures = []
    costs_sma_201 = []
    costs_sma_5 = []

    MAX_ITERATIONS = 1_000_000

    # SA Variables
    # collegare temteratura all'andamento della qualità delle soluzioni
    TIME_OUT = 4000 # in seconds

    TEMPERATURE = 100.0
    FREEZING_TEMPERATURE = 0.001

    TEMPERATURE_DECREASE_STEP_1 = 0.95
    TRESHOLD_DECREASE_STEP_1 = 1
    TEMPERATURE_DECREASE_STEP_2 = 0.99

    TEMPERATURE_REFUEL_STUCK = 60
    TEMPERATURE_REFUEL = 1

    ITEMS_PER_TEMPERATRE = 1

    # holds the SMA of the last 50 accepted solutions costs
    SMA_COST_201 = IntegerCircularBuffer(201)
    SMA_COST_5 = IntegerCircularBuffer(5)

    print("*** START SA ROUTINE ***\n")
    start = time.process_time()

    # Generate starting random solution
    curr_node_pe, curr_pe_nodes = random_sol_generator(schedule, size_x, size_y)
    node_pe = curr_node_pe
    pe_nodes = curr_pe_nodes
    sol_cost, sol_worst_nodes = cost_space_solution(node_pe, dfg, arch, size_x, size_y)
    SMA_COST_5.next(sol_cost)
    SMA_COST_201.next(sol_cost)

    items = 0
    refuels = 0
    iterations = 0

    while sol_cost != 0 and iterations < MAX_ITERATIONS:
        if TIME_OUT < time.process_time() - start:
            print("TIMED OUT")
            break
        # print(f"T: {TEMPERATURE}")
        # print(f"sol_worst_nodes: {sol_worst_nodes}")

        # add best pe nodes here
        curr_node_pe, curr_pe_nodes = neighbour_sol_generator(schedule, node_pe, sol_worst_nodes, size_x, size_y)
        c, curr_sol_worst_nodes = cost_space_solution(curr_node_pe, dfg, arch, size_x, size_y)
        # print(f"New solution cost: {c}")

        if c < sol_cost:
            node_pe = curr_node_pe
            pe_nodes = curr_pe_nodes
            sol_cost = c
            sol_worst_nodes = curr_sol_worst_nodes
            # print(f"New solution is the best solution found so far, keeping it")
        else:
            # apply boltzman
            delta_E = c - sol_cost
            P = math.exp(- delta_E / TEMPERATURE)
            rnd = random.random()
            # print(f"Delta E: {delta_E}, probability: {P}, random: {rnd}")

            if rnd < P:
                # print(f"New solution is worse but we accept it: hill climb")
                node_pe = curr_node_pe
                pe_nodes = curr_pe_nodes
                sol_cost = c
                sol_worst_nodes = curr_sol_worst_nodes

        costs.append(sol_cost)
        temperatures.append(TEMPERATURE)
        costs_sma_201.append(SMA_COST_201.sma_prev)
        costs_sma_5.append(SMA_COST_5.sma_prev)

        if items >= ITEMS_PER_TEMPERATRE - 1:
            # track what we keep from these items within the single temperature step
            SMA_COST_5.next(sol_cost)
            SMA_COST_201.next(sol_cost)
            items = 0

            if TEMPERATURE < TRESHOLD_DECREASE_STEP_1:
                TEMPERATURE = TEMPERATURE * TEMPERATURE_DECREASE_STEP_2
            else: 
                TEMPERATURE = TEMPERATURE * TEMPERATURE_DECREASE_STEP_1

            # if we freeze we want to make sure that the cost has not been downtrending
            if TEMPERATURE < FREEZING_TEMPERATURE:
                if SMA_COST_5.sma_prev < SMA_COST_201.sma_prev:
                    # we are downtrending so we should keep exploring
                    # with low temperature not to go too far around
                    # and distrup previous work
                    if refuels > 3:
                        TEMPERATURE += TEMPERATURE_REFUEL_STUCK
                        refuels = 0
                    else:
                        TEMPERATURE += TEMPERATURE_REFUEL
                        refuels += 1
        else:
            items += 1

        iterations += 1
    end = time.process_time()
    total_time = end - start

    print("\n*** END SA ROUTINE ***\n")
    print(f"End solution cost: {sol_cost}")

    # Plot search data
    if BENCHMARK:
        file_path = saveFigTemperature(iterations, temperatures, costs, costs_sma_201, costs_sma_5, BENCHMARK.get_directory_path_str(), BENCHMARK.sa_algorithm_type, BENCHMARK.id)
        BENCHMARK.save_results(total_time, sol_cost, iterations, file_path)
    else:
        saveFigTemperature(iterations, temperatures, costs, costs_sma_201, costs_sma_5)
    
    print("\n\n\n")
    return (node_pe, pe_nodes, total_time)
