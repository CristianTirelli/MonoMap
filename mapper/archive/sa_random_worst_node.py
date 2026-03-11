
import math
import random
import time

from networkx import DiGraph, Graph, shortest_path_length

from plots import saveFigTemperature

# Not working

# ALGORITHM:    The overall algorithm is a classica sa, we loop until the temperature freezes or we
#               reach a valid solution. We initialize the first solution at random and compute its cost.
#               Then from that solution we compute a NEIGHBOUR solution and evaluate it, if it is better
#               we keep it and if it is worse we randomly select based on temperature and cost delta if we
#               keep it (hill climb) or not.

# TEMPERATURE:  The simulated annealing temperature is is detached from the algorithm performances or
#               the problem variables. The temperature decreases from 100 to 0.001 in two steps by 0.95
#               each time as it is greater then 0.5 then it decreases by 0.99. Values are arbitrary. We
#               hold a variable that holds how many solution per each temperature level we want to search
#               before updating the temperature, it usually fluctuates betwee 1 and 5.

# COST:         The cost is computed by looping over all dependency edges and check if they are respected,
#               if they are not we add the distance squared to the cost to discourage further away dependencies.
#               While we compute the cost we keep the worst node that is in the current solution. By worst
#               node we mean, we keep track of the worst edge in the current solution (the one with the
#               furthest) distance and we then chose at random one between the source and the destination
#               node, which becomes the new returned worst node tha will then be used by NEIGHBOUR.

# NEIGHBOUR:    The routine that computes the neighbour solution is as follows: We take the schedule, the
#               previous solution and the worst node computed from COST. We copy over all operations PEs
#               poisitons of schedules that do not have the worst node present and all operations that are
#               in the same schedule of the worst node but that are not the worst node. Then out of all
#               available PEs position we select a new random position for the worst node. 


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


def neighbour_sol_generator(schedule: dict[str, list[int]], from_node_pe: dict[int, int], from_pe_nodes: dict[int, list[int]], from_worst_node: int, size_x: int, size_y: int) -> tuple[dict[int, int], dict[int, list[int]]]:
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

    size = size_x * size_y

    new_pe: int
    # init already without current pe position
    available_pes: list[int] = [i for i in range(size)]

    # identfy which pe is worst node from
    for pe in from_pe_nodes:
        if from_worst_node in from_pe_nodes[pe]:
            # generate a random position for this node
            from_pe_nodes[pe].remove(from_worst_node)

            # find who's at the same time
            for t in schedule:
                if from_worst_node in schedule[t]:
                    # n are all operations scheduled at the same time, also itself
                    for n in schedule[t]:
                        available_pes.remove(from_node_pe[n])
            break

    new_pe = random.choice(available_pes)
    # print(new_pe)
    # add to pe_nodes and node_pe
    if new_pe not in from_pe_nodes:
        from_pe_nodes[new_pe] = []
    from_pe_nodes[new_pe].append(from_worst_node)
    from_node_pe[from_worst_node] = new_pe
    return (from_node_pe, from_pe_nodes)


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


def cost_space_solution(pe_nodes: dict[int, int], dfg: DiGraph, arch: Graph, size_x: int, size_y: int) -> tuple[int, int]:
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

    # The function has to look at all edges to determine cost

    cost = 0
    max_distance: int = -1
    worst_node: int = -1

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

                            if max_distance < 0:
                                max_distance = d
                                worst_node = random.choice([source, destination])
                            else:
                                if max_distance < d:
                                    max_distance = d
                                    worst_node = random.choice([source, destination])
    return cost, worst_node


def simulatedAnnealingSearch(schedule: dict[str, list[int]], dfg: DiGraph, arch: Graph, II: int, size_x: int, size_y: int) -> tuple[dict[int, int],  dict[int, list[int]], float]:
    print("\n\n\n")
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}
    sol_cost: int
    worst_node: int

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
    sol_cost, worst_node = cost_space_solution(pe_nodes, dfg, arch, size_x, size_y)

    items = 0

    # while FREEZING_TEMPERATURE < TEMPERATURE and check_solution(pe_nodes, dfg, size_y, size_x) == False:
    while FREEZING_TEMPERATURE < TEMPERATURE and sol_cost != 0:
        # print(f"T: {TEMPERATURE}")

        # add best pe nodes here
        curr_node_pe, curr_pe_nodes = neighbour_sol_generator(schedule, node_pe, pe_nodes, worst_node, size_x, size_y)
        c, curr_worst_node = cost_space_solution(curr_pe_nodes, dfg, arch, size_x, size_y)
        # print(f"New solution cost: {c}")

        if c < sol_cost:
            node_pe = curr_node_pe
            pe_nodes = curr_pe_nodes
            sol_cost = c
            worst_node = curr_worst_node

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
                worst_node = curr_worst_node

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
