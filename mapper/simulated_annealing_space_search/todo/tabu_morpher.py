import random
from dataclasses import dataclass, field

from networkx import Graph, shortest_path_length

from benchmark import Benchmark

# TODO unconnected old implementation class

# ALGORITHM:    We have a max cap of iterations 1_000_000 and a temperature that is detached from the main loop,
#               until we reach a solution, we complete all iterations or we timeout we continue the search.

# TEMPERATURE:  Temperature moves down following Morpher cooling schedule, and SMA: 201 and 5, if 201 reaches 5
#               we warm up following a schedule: += 0.001 / math.log(warming_up_step + math.e) then if we find a
#               better solution or we reach a cap of 1000 times the freezing temperature we stop warming up and
#               continue cooling

# COST:         The cost function simply calculates the Manhattan distance of nodes that do not resespect dependencies

@dataclass
class TabuMorpher:
    TEMPERATURE_STRATEGY_ID: str = field(default="MORPHER-SMA-BEST-AND-MAX-CAP-TABU_", init=False)

    # Runtime variables
    # holds the current best solution
    node_pe: dict[int, int] = field(default_factory=dict, init=False)
    pe_nodes: dict[int, list[int]] = field(default_factory=dict, init=False)
    cost: int = field(default=0, init=False)

    temperature: float = field(default=0.0, init=False)

    cost_sma_slow: float = field(default=0.0, init=False)
    cost_sma_fast: float = field(default=0.0, init=False)

    ## COSTANTS ##
    # input specific data
    schedule: dict[str, list[int]]
    size_x: int
    size_y: int

    dfg: Graph
    arch: Graph

    BENCHMARK: Benchmark

    # iterations and timeout
    MAX_ITERATIONS: int = field(default=1_000_000, init=False)
    TIME_OUT: int = field(default=60 * 50, init=False)
    # TIME_OUT: int = field(default=4000, init=False)

    # temperature
    START_TEMPERATURE: int = field(default=100, init=False)
    FREEZING_TEMPERATURE: float = field(default=0.01, init=False)
    TEMPERATURE_REFUEL_LIMIT: float = field(default=3, init=False)

    ITEMS_PER_TEMPERATRE: int = field(default=50, init=False)

    # SMA
    SMA_SLOW_ITEMS: float = field(default=201, init=False)
    SMA_FAST_ITEMS: float = field(default=5, init=False)

    EPSILON: float = field(default=0.001, init=False)

    # Shared functions
    def random_sol_generator(self):
        """
        Random solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.
        """
        node_pe: dict[int, int] = {}
        pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        for t in self.schedule:
            available_pes = [i for i in range(size)]

            for node in self.schedule[t]:
                pe = random.choice(available_pes)
                
                if pe not in pe_nodes:
                    pe_nodes[pe] = []
                pe_nodes[pe].append(node)

                if node not in node_pe:
                    node_pe[node] = pe
                else:
                    print(f"should not happend, node: {node}, pe: {pe}")

                available_pes.remove(pe)

        # apply to main variables
        self.node_pe = node_pe
        self.pe_nodes = pe_nodes

    @staticmethod
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

    @staticmethod
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

    # Implementation Specific (Strategy)
    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        The implementation differs from strategy to strategy

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """
        raise NotImplementedError
    
    # Semi-Hybrid: Some strategies may add extra computation to the cost function
    def cost_space_solution(self, curr_node_pe: dict[int, int]) -> int:
        """
        We need to add a value, a "cost" to all actions that randomness produces and that bring us away from a valid solution.
        We need to structure those costs in such a way that the more wrong the action is the more it costs.
        Wrong actions are, in order of more bad to less bad:

        Schedule two instructions on the same PE at the same scheduling time (Impossible as for our current construction)
        Do not respect dependencies, changes based on how far the two dependent instructions are

        So the lower the cost the better solution it is (near to a valid solution)

        We assume solutions that reach a cost of 0 (valid solution) are all equally good.

        :return: The cost of the current solution. A cost of zero indicates that the solution is valid
        :rtype: int
        """
        cost = 0

        for e in self.dfg.edges:
            source = e[0]
            destination = e[1]

            # should always be present
            if source not in curr_node_pe:
                raise AssertionError("source is not present in node_pe")
            if destination not in curr_node_pe:
                raise AssertionError("destination is not present in node_pe")
            ps = curr_node_pe[source]
            pd = curr_node_pe[destination]

            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x):
                cost += self.pe_distance(self.arch, pd, ps) ** 2
        return cost

    def updateTemperature(self, t: float, acceptance_rate: float) -> int:
        if acceptance_rate > 0.96:
            return t * 0.5
        elif acceptance_rate > 0.8:
            return t * 0.9
        elif acceptance_rate > 0.15:
            return t * 0.98
        else:
            return t * 0.95

    # Hybrid: Overall Temperature and Simulated Annealing search is shared
    def simulatedAnnealingSearch(self) -> tuple[dict[int, int],  dict[int, list[int]], float]:
        """
        Tabu specific SA search is overridden by strategy
        """
        raise NotImplementedError
