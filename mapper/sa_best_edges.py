import math
import random
import time

from networkx import DiGraph, Graph, shortest_path_length

from plots import saveFigTemperature

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


def neighbour_sol_generator(schedule: dict[str, list[int]], from_node_pe: dict[int, int], keep_nodes: list[int], size_x: int, size_y: int) -> tuple[dict[int, int], dict[int, list[int]]]:
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

    for n in keep_nodes:
        pe = from_node_pe[n]

        if pe not in pe_nodes:
            pe_nodes[pe] = []
        pe_nodes[pe].append(n)

        node_pe[n] = pe

    for t in schedule:
        # keep only unassigned operations
        available_nodes = [op for op in schedule[t] if op not in keep_nodes]

        # keep only available PEs
        available_pes = [i for i in range(size)]

        for node in schedule[t]:
            if node in keep_nodes:
                available_pes.remove(node_pe[node])

        for av_node in available_nodes:
            pe = random.choice(available_pes)
            
            if pe not in pe_nodes:
                pe_nodes[pe] = []
            pe_nodes[pe].append(av_node)

            if av_node not in node_pe:
                node_pe[av_node] = pe
            else:
                print(f"should not happend, node: {av_node}, pe: {pe}")

            available_pes.remove(pe)
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


def cost_space_solution(pe_nodes: dict[int, int], schedule: dict[str, list[int]], dfg: DiGraph, arch: Graph, size_x: int, size_y: int) -> tuple[int, list[int]]:
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
    :type pe_nodes: dict[int, int]

    :param dfg: The DFG graph dependencies
    :type dfg: DiGraph

    :return: The cost of the current solution. A cost of zero indicates that the solution is valid
    :rtype: int
    """
    # schedule would be needed in the case that operations can be sceduled at times t+k, k \in N,
    # i.e. there may be more operations that CGRA size

    cost = 0

    # we keep the best 30% edges
    BEST_EDGES_TO_KEEP = 0.57
    lcdn: list[tuple[int, int, int]] = []
    n_best = math.floor(len(list(dfg.edges)) * BEST_EDGES_TO_KEEP)

    for e in dfg.edges:
        source = e[0]
        destination = e[1]
        for ps in pe_nodes:
            if source in pe_nodes[ps]:
                for pd in pe_nodes:
                    if destination in pe_nodes[pd]:
                        d = 0
                        if not isConnected(pd, ps, size_y, size_x):
                            # if not connected means that pd, ps are int that they are not adjacent
                            d = pe_distance(arch, pd, ps)
                            # print(f"Source operation {source} at pe {pd} to operation {destination} at pe {ps} has distance {d}")
                            cost += d ** 2

                        # keep unordered n_best
                        if len(lcdn) < n_best:
                            lcdn.append([d, source, destination])
                        else:
                            highest_d = lcdn[0][0]
                            highest_d_idx = 0
                            for i in range(1, len(lcdn)):
                                if highest_d < lcdn[i][0]:
                                    highest_d = lcdn[i][0]
                                    highest_d_idx = i

                            if d < highest_d:
                                lcdn[highest_d_idx] = [d, source, destination]
    
    # unwrap
    nodes = []
    for _, src, dest in lcdn:
        if src not in nodes:
            nodes.append(src)
        if dest not in nodes:
            nodes.append(dest)

    return cost, nodes


def simulatedAnnealingSearch(schedule: dict[str, list[int]], dfg: DiGraph, arch: Graph, II: int, size_x: int, size_y: int) -> tuple[dict[int, int],  dict[int, list[int]], float]:
    print("\n\n\n")
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}
    sol_cost: int
    keep_nodes = []

    # Data to study the search
    costs = []
    temperatures = []

    # SA Variables
    TEMPERATURE = 100.0
    FREEZING_TEMPERATURE = 0.001

    TEMPERATURE_DECREASE_STEP_1 = 0.95
    TRESHOLD_DECREASE_STEP_1 = 0.5

    TEMPERATURE_DECREASE_STEP_2 = 0.99
    
    ITEMS_PER_TEMPERATURE = 5

    # What can we exploit during the search phase?
    # -> Scheduling has to be respected
    # Q: is it possible to recieve more operations than PEs at a given time schedule t?
    # -> It does not matter if there are operations on the same PEs between dfg dependencies scheduled at different clock times
    # see Q: at check_solution

    print("*** START SA ROUTINE ***\n")
    start = time.time()

    # Generate starting random solution
    curr_node_pe, curr_pe_nodes = random_sol_generator(schedule, size_x, size_y)
    node_pe = curr_node_pe
    pe_nodes = curr_pe_nodes
    sol_cost, keep_nodes = cost_space_solution(pe_nodes, schedule, dfg, arch, size_x, size_y)

    items = 0

    # while FREEZING_TEMPERATURE < TEMPERATURE and check_solution(pe_nodes, dfg, size_y, size_x) == False:
    while FREEZING_TEMPERATURE < TEMPERATURE and sol_cost != 0:
        # print(f"T: {TEMPERATURE}")

        # add best pe nodes here
        curr_node_pe, curr_pe_nodes = neighbour_sol_generator(schedule, node_pe, keep_nodes, size_x, size_y)
        c, curr_keep_nodes = cost_space_solution(curr_pe_nodes, schedule, dfg, arch, size_x, size_y)

        # print(f"New solution cost: {c}")

        if c < sol_cost:
            node_pe = curr_node_pe
            pe_nodes = curr_pe_nodes
            sol_cost = c
            keep_nodes = curr_keep_nodes
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
                keep_nodes = curr_keep_nodes

        costs.append(sol_cost)
        temperatures.append(TEMPERATURE)

        # decrease temperature
        if items == ITEMS_PER_TEMPERATURE:
            if TEMPERATURE > TRESHOLD_DECREASE_STEP_1:
                TEMPERATURE = TEMPERATURE * TEMPERATURE_DECREASE_STEP_1
            else:
                TEMPERATURE = TEMPERATURE * TEMPERATURE_DECREASE_STEP_2
            items = 0
        else:
            items += 1
        # print()
    end = time.time()

    print("\n*** END SA ROUTINE ***\n")
    print(f"End solution cost: {sol_cost}")

    # Plot search data
    saveFigTemperature(temperatures, costs)
    
    print("\n\n\n")
    return (node_pe, pe_nodes, end - start)
