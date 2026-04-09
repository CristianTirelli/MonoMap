
import math
import random
import time

from networkx import DiGraph, Graph, shortest_path_length

from plots import saveFigTemperature


# ALGORITHM:    The overall algorithm is a classica sa, we loop until the temperature freezes or we
#               reach a valid solution. We initialize the first solution at random and compute its cost.
#               Then from that solution we compute a NEIGHBOUR solution and evaluate it, if it is better
#               we keep it and if it is worse we randomly select based on temperature and cost delta if we
#               keep it (hill climb) or not.

# TEMPERATURE:  The simulated annealing temperature is is detached from the algorithm performances or
#               the problem variables. The temperature decreases from 100 to 0.001 in two steps by 0.95
#               each time as it is greater than 50 then it decreases by 0.99. Values are arbitrary. We
#               hold a variable that holds how many solution per each temperature level we want to search
#               before updating the temperature, it usually fluctuates betwee 1 and 5.

# COST:         The cost is computed by looping over all dependency edges and check if they are respected,
#               if they are not we add the distance squared to the cost to discourage further away dependencies.

# NEIGHBOUR:    The routine that computes the neighbour solution is as follows: We take the schedule and the
#               previous solution. We then randomly select a node between all available nodes within the DFG
#               by random it means that there is no heuristic in the selection. We then fix all other nodes and
#               we randomly select a new placement for that previously picked node between all available PEs
#               its previous placement PE excluded.


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


def neighbour_sol_generator(schedule: dict[str, list[int]], from_node_pe: dict[int, int], size_x: int, size_y: int) -> tuple[dict[int, int], dict[int, list[int]]]:
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

    node_to_move = random.randint(0, len(from_node_pe))

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


def cost_space_solution(pe_nodes: dict[int, list[int]], dfg: DiGraph, arch: Graph, size_x: int, size_y: int) -> int:
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
    # schedule would be needed in the case that operations can be sceduled at times t+k, k \in N,
    # i.e. there may be more operations that CGRA size
    cost = 0

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
    return cost


def simulatedAnnealingSearch(schedule: dict[str, list[int]], dfg: DiGraph, arch: Graph, II: int, size_x: int, size_y: int) -> tuple[dict[int, int],  dict[int, list[int]], float]:
    print("\n\n\n")
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}
    sol_cost: int

    # Data to study the search
    costs = []
    temperatures = []

    # SA Variables
    # collegare temteratura all'andamento della qualità delle soluzioni
    TEMPERATURE = 100.0

    FREEZING_TEMPERATURE = 0.001
    TEMPERATURE_DECREASE_STEP_1 = 0.95
    TRESHOLD_DECREASE_STEP_1 = 50
    TEMPERATURE_DECREASE_STEP_2 = 0.99

    ITEMS_PER_TEMPERATURE = 5

    # acceptance_rate = 0

    print("*** START SA ROUTINE ***\n")
    start = time.process_time()

    # Generate starting random solution
    curr_node_pe, curr_pe_nodes = random_sol_generator(schedule, size_x, size_y)
    node_pe = curr_node_pe
    pe_nodes = curr_pe_nodes
    sol_cost = cost_space_solution(pe_nodes, dfg, arch, size_x, size_y)

    items = 0

    # while FREEZING_TEMPERATURE < TEMPERATURE and check_solution(pe_nodes, dfg, size_y, size_x) == False:
    while FREEZING_TEMPERATURE < TEMPERATURE and sol_cost != 0:
        # print(f"T: {TEMPERATURE}")

        # add best pe nodes here
        curr_node_pe, curr_pe_nodes = neighbour_sol_generator(schedule, node_pe, size_x, size_y)
        c = cost_space_solution(curr_pe_nodes, dfg, arch, size_x, size_y)

        # print(f"New solution cost: {c}")

        if c < sol_cost:
            node_pe = curr_node_pe
            pe_nodes = curr_pe_nodes
            sol_cost = c
            # print(f"New solution is the best solution found so far, keeping it")
            # acceptance_rate += 1
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
                # acceptance_rate += 1

        costs.append(sol_cost)
        temperatures.append(TEMPERATURE)

        # decrease temperature
        if items == ITEMS_PER_TEMPERATURE:
            # TEMPERATURE = updateTemperature(TEMPERATURE, acceptance_rate / items)
            if TEMPERATURE > TRESHOLD_DECREASE_STEP_1:
                TEMPERATURE = TEMPERATURE * TEMPERATURE_DECREASE_STEP_1
            else:
                TEMPERATURE = TEMPERATURE * TEMPERATURE_DECREASE_STEP_2
            items = 0
        else:
            items += 1

        # print()
    end = time.process_time()

    print("\n*** END SA ROUTINE ***\n")
    print(f"End solution cost: {sol_cost}")

    # Plot search data
    saveFigTemperature(temperatures, costs)
    
    print("\n\n\n")
    return (node_pe, pe_nodes, end - start)


def updateTemperature(t: float, acceptance_rate: float):
	if acceptance_rate > 0.96:
		return t * 0.99
	elif acceptance_rate > 0.8:
		return t * 0.98
	elif acceptance_rate > 0.15:
		return t * 0.95
	else:
		return t * 0.5
