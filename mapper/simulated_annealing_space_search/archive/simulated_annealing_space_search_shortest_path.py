import math
import random
import time
from dataclasses import dataclass, field

from networkx import Graph, shortest_path_length

from benchmark import Benchmark
from plots import save_plot_run_graphs

# Base SA class that implementes, collects and prepare all methods and data used by strategies
# Then strategies have two components: Temperature schedule, which decides how the main SA loop
# is managed, and Neighbor schedule, which dictates how the neighbor solution is constructed.

@dataclass
class SimulatedAnnealingSpaceSearch:
    # No id as it is inherited by any strategy class
    # and does not determine any difference for a class



    # Runtime variables
    # holds the current best solution
    node_pe: dict[int, int] = field(default_factory=dict, init=False)
    pe_nodes: dict[int, list[int]] = field(default_factory=dict, init=False)
    cost: int = field(default=0, init=False)

    start_configuration_cost: int = field(default=0, init=False)

    # holds randomness seeds
    seed_start_configuration: float = field(default=12345, kw_only=True)
    seed_algorithm_run: float = field(default=None, kw_only=True)

    # holds random generators
    randgen_start_configuration: random.Random = field(default=None, init=False)
    randgen_algorithm_run: random.Random = field(default=None, init=False)

    temperature: float = field(default=0.0, init=False)

    cost_sma_slow: float = field(default=0.0, init=False)
    cost_sma_fast: float = field(default=0.0, init=False)

    iterations: int = field(default=0, init=False)

    start_time: float = field(default=0.0, init=False)

    # holds the (pe1, pe2) = hops distances of two pes
    pe_distance_cache: dict[tuple[int, int], int] = field(default_factory=dict, init=False)



    ## COSTANTS ##
    # input specific data
    schedule: dict[str, list[int]]
    size_x: int
    size_y: int

    dfg: Graph
    arch: Graph

    BENCHMARK: Benchmark

    # may be helpful have dict: (node, time) = pe

    # holds pe -> schedule time
    node_schedule_t: dict[int, int] = field(default_factory=dict, init=False)

    # number of nodes
    nnodes: int = field(default=0, init=False)

    # iterations and timeout
    MAX_ITERATIONS: int = field(default=1_000_000, init=False)
    TIME_OUT: int = field(default=4000, init=False)

    # temperature
    START_TEMPERATURE: int | None = field(default=None, init=False)
    ITEMS_PER_TEMPERATRE: int = field(default=0, init=False)
    FREEZING_TEMPERATRE: float = field(default=0.001, init=False)

    # SMA
    SMA_SLOW_ITEMS: float = field(default=100, init=False)
    SMA_FAST_ITEMS: float = field(default=5, init=False)

    EPSILON: float = field(default=0.001, init=False)



    ## BENCHMARK DS ##
    ## Data that is colleced each run if benchmarking ##
    costs: list[float] = field(default_factory=list, init=False)
    temperatures: list[float] = field(default_factory=list, init=False)
    probabilities: list[float] = field(default_factory=list, init=False)
    costs_sma_fast: list[float] = field(default_factory=list, init=False)
    costs_sma_slow: list[float] = field(default_factory=list, init=False)

    total_items_iterations: int = field(default=0, init=False)
    cumulative_neighbor_sol_time_item: float = field(default=0.0, init=False)
    cumulative_cost_space_sol_time_item: float = field(default=0.0, init=False)

    cumulative_sol_check_items_routine_time: float = field(default=0.0, init=False)

    cumulative_temp_routine_time: float = field(default=0.0, init=False)
    cumulative_running_time: float = field(default=0.0, init=False)



    def __post_init__(self):
        print(f"Initial solution of seed: {self.seed_start_configuration}")
        self.randgen_start_configuration = random.Random(self.seed_start_configuration)

        if self.seed_algorithm_run:
            self.randgen_algorithm_run = random.Random(self.seed_algorithm_run)
        else:
            # arbitrary large random seed
            self.seed_algorithm_run = random.randrange(10**8)
            self.randgen_algorithm_run = random.Random(self.seed_algorithm_run)

        # compute dfg number of nodes
        self.nnodes = len(list(self.dfg))
 
        # build node_schedule_t
        for t in self.schedule:
            for n in self.schedule[t]:
                self.node_schedule_t[n] = t

        # compute PEs distances if they are not connected
        all_pes = list(range(self.size_x * self.size_y))
        for pe1 in all_pes:
            for pe2 in all_pes:
                if not self.isConnected(pe1, pe2, self.size_y, self.size_x):
                    self.pe_distance_cache[(pe1, pe2)] = shortest_path_length(self.arch, source=pe1, target=pe2)
                else:
                    self.pe_distance_cache[(pe1, pe2)] = 0

        # Generate starting random solution and evaluate it
        self.initial_sol_generator()

        # compute solution cost
        self.sol_cost = self.cost_space_solution(self.node_pe)
        self.start_configuration_cost = self.sol_cost

        # init MA
        self.cost_sma_fast = float(self.sol_cost)
        self.cost_sma_slow = float(self.sol_cost)

        print(f"Algorithm run seed: {self.seed_algorithm_run}")

    @staticmethod
    def isConnected(pe1: int, pe2: int, size_y: int, size_x: int, debug = False) -> bool:
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
                if debug:
                    print(f"pe1: {pe1}, pe2: {pe2} is connected same row")
                return True
            if abs(pe1 - pe2) == size_y - 1:
                if debug:
                    print(f"pe1: {pe1}, pe2: {pe2} is connected same row around")
                return True

        # same col
        if j1 == j2:
            if (pe1 == pe2 + size_y) or (pe1 == pe2 - size_y):
                if debug:
                    print(f"pe1: {pe1}, pe2: {pe2} is connected same col")
                return True
            if abs(i1 - i2) == size_x - 1:
                if debug:
                    print(f"pe1: {pe1}, pe2: {pe2} is connected same col around")
                return True

        # center
        if pe1 == pe2:
            if debug:
                print(f"pe1: {pe1}, pe2: {pe2} is connected on same pe")
            return True
        
        if debug:
            print(f"pe1: {pe1}, pe2: {pe2} is not connected rows i1: {i1} i2: {i2} columns, j1: {j1} j2: {j2}")
        return False

    def pe_distance(self, pe1: int, pe2: int) -> int:
        """
        Computes the distance from `pe1` to `pe2` in number of hops (edges) from source to target, respectively

        It assumes pe1 and pe2 are are already sized to the architecture, i.e. [0, size)

        :param pe1: The source PE
        :type pe1: int

        :param pe2: The target PE
        :type pe2: int

        :return: The distance from pe1 to pe2 in number of edges
        :rtype: int
        """
        return self.pe_distance_cache[(pe1, pe2)]

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

    # Implementation Specific (Temperature Routine)
    def temperature_routine(self):
        """
        Temperature Routine procedure

        Responsible of handling:
        
            - search routine
            - add all benchmarking data to relative variables
        """
        raise NotImplementedError
    
    # Semi-Hybrid: Some strategies may add extra computation to the cost function
    def cost_space_solution(self, curr_node_pe: dict[int, int], p: str | None = None) -> int:
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

        if p:
            print(f"size_x: {self.size_x}, size_y: {self.size_y}")

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

            if p:
                print(f"{p} Check: source, pe: {source}, {ps} destination, pe: {destination}, {pd} connected: {self.isConnected(pd, ps, self.size_y, self.size_x)}")

            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x, True if p is not None else False):
                cost += self.pe_distance(pd, ps) ** 2
        
        if p:
            print(f"{p} final cost: {cost}")
        return cost

    # Hybrid Functions
    # Generation of an initial solution
    def initial_sol_generator(self):
        """
        Initial solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.

        This implementation generates a random initial solution
        """
        node_pe: dict[int, int] = {}
        pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        for t in sorted(self.schedule):
            available_pes = [i for i in range(size)]

            for node in sorted(self.schedule[t]):
                pe = self.randgen_start_configuration.choice(available_pes)
                
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

    # Overall Temperature and Simulated Annealing search is shared
    def simulatedAnnealingSearch(self) -> tuple[dict[int, int],  dict[int, list[int]], float]:
        if not self.START_TEMPERATURE:
            raise AssertionError("Start temperature must be set before calling simulatedAnnealingSearch")
        self.temperature = self.START_TEMPERATURE

        print()
        print("*** START SA ROUTINE ***\n")
        self.start_time = time.process_time()
        self.temperature_routine()
        total_time = time.process_time() - self.start_time
        print("\n*** END SA ROUTINE ***\n")

        print(f"End solution cost: {self.sol_cost}")
        print(f"Total time: {total_time}")
        print(f"Total iterations: {self.iterations}")
        print(f"Start configuration cost: {self.start_configuration_cost} and seed: {self.seed_start_configuration}")

        # Now we collect extra data from the run:
        # number of nodes that are positioned incorrectly and correctly
        # number for both and list for incorrect positioning
        incorrect_source_destination_nodes: dict[int, int] = {}

        # node and relative distance cost
        incorrect_node_cost: dict[tuple[int, int], int] = {}
        incorrectly_positioned_nodes_list: list[int] = []

        for e in self.dfg.edges:
            source: int = e[0]
            destination: int = e[1]

            ps = self.node_pe[source]
            pd = self.node_pe[destination]

            if not self.isConnected(pd, ps, self.size_y, self.size_x):
                inc = self.pe_distance(pd, ps) ** 2
                incorrect_node_cost[(source, destination)] = inc
                incorrect_source_destination_nodes[source] = destination

                if destination not in incorrectly_positioned_nodes_list:
                    incorrectly_positioned_nodes_list.append(destination)
                if source not in incorrectly_positioned_nodes_list:
                    incorrectly_positioned_nodes_list.append(source)

        # number of incorrectly positioned nodes
        incorrectly_positioned_nodes: int = len(incorrectly_positioned_nodes_list)

        # Plot search data
        plot_files_path = save_plot_run_graphs(
            # cgra plot
            self.node_pe,
            self.schedule,
            self.size_x,
            self.size_y,

            # t plots
            self.iterations,
            self.temperatures,
            self.probabilities,
            self.costs,

            # benchamrk
            self.BENCHMARK.get_directory_path_str() if self.BENCHMARK else None,
            self.BENCHMARK.id if self.BENCHMARK else None,
                
            temp_cost={
                "costs_sma_slow": self.costs_sma_slow,
                "costs_sma_fast": self.costs_sma_fast
            }
        )

        if self.BENCHMARK:
            self.BENCHMARK.save_results(
                total_time,
                self.sol_cost,
                self.start_configuration_cost,
                self.iterations,
                self.ITEMS_PER_TEMPERATRE,
                plot_files_path,
                len(list(self.dfg)) - incorrectly_positioned_nodes,
                incorrectly_positioned_nodes,
                incorrect_source_destination_nodes,
                incorrect_node_cost,
                self.seed_start_configuration,
                self.seed_algorithm_run,

                average_neighbor_sol_time_item =(self.cumulative_neighbor_sol_time_item / self.total_items_iterations),
                average_cost_space_sol_time_item =(self.cumulative_cost_space_sol_time_item / self.total_items_iterations),
                average_sol_check_items_routine_time =(self.cumulative_sol_check_items_routine_time / self.iterations),
                average_temp_routine_time =(self.cumulative_temp_routine_time / self.iterations),
                average_running_time =(self.cumulative_running_time / self.iterations)
            )

        print()
        return (self.node_pe, self.pe_nodes, total_time)
