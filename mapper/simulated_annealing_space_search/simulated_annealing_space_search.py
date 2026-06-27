import random
import time
from dataclasses import dataclass, field

from networkx import DiGraph, Graph

from benchmark import Benchmark
from recorder import Recorder
from plots import save_plot_run_graphs

# Base SA class that implementes, collects and prepare all methods and data used by strategies
# Then strategies have two components: Temperature schedule, which decides how the main SA loop
# is managed, and Neighbor schedule, which dictates how the neighbor solution is constructed.

@dataclass
class SimulatedAnnealingSpaceSearch:
    # Define all IDs here so that it is cleaner
    ID: str = field(default=None, init=False)

    START_ID: str = field(default=None, init=False)
    STRATEGY_ID: str = field(default=None, init=False)
    ROUTINE_ID: str = field(default=None, init=False)


    ## Input ##
    # input specific data
    schedule: dict[str, list[int]]
    size_x: int
    size_y: int

    directed_dfg: DiGraph
    dfg: Graph
    arch: Graph

    BENCHMARK: Benchmark
    RECORDER: Recorder


    ## Runtime variables ##
    # holds the current best solution
    node_pe: dict[int, int] = field(default_factory=dict, init=False)
    pe_nodes: dict[int, list[int]] = field(default_factory=dict, init=False)
    cost: int = field(default=0, init=False)

    # holds the current neighbouring solutions, it is the data structure used
    # by neighbour_sol_generator and undo_neighbour_sol_generator to 
    curr_node_pe: dict[int, int] = field(default_factory=dict, init=False)

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

    MODEL_NUMBER: int = field(default=None, kw_only=True)

    # may be helpful have dict: (node, time) = pe, but as to be bookept: runtime variable


    ## CONSTANTS ##
    # holds node -> schedule time
    NODE_SCHEDULE_T: dict[int, int] = field(default_factory=dict, init=False)

    # number of nodes
    NNODES: int = field(default=0, init=False)

    # iterations and timeout
    MAX_ITERATIONS: int = field(default=1_000_000, init=False)
    TIME_OUT: int = field(default=4000, init=False)

    # temperature
    START_TEMPERATURE: int | None = field(default=None, init=False)
    ITEMS_PER_TEMPERATRE: int = field(default=0, init=False)
    FREEZING_TEMPERATRE: float = field(default=0.001, init=False)
    START_TEMPERATURE_COEFF: int = field(default=10, kw_only=True)

    # SMA
    SMA_SLOW_ITEMS: float = field(default=100, init=False)
    SMA_FAST_ITEMS: float = field(default=5, init=False)

    EPSILON: float = field(default=0.001, init=False)

    SMA_REHEATING_THRESHOLD_PERCENTAGE: float = field(default=0.001, init=False)
    ACCEPTANCE_RATE_REHEATING_THRESHOLD_PERCENTAGE: float = field(default=1e-6, init=False)


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


    # In constructors: It is assumed we work only on static data, passed at initialization
    # you can't assume that Runtime variables are ready, it is a place to initialize COSTANTS
    def __post_init__(self):
        self.randgen_start_configuration = random.Random(self.seed_start_configuration)
        print(f"Initial solution of seed: {self.seed_start_configuration}")

        if self.seed_algorithm_run:
            self.randgen_algorithm_run = random.Random(self.seed_algorithm_run)
        else:
            # arbitrary large random seed
            self.seed_algorithm_run = random.randrange(10**8)
            self.randgen_algorithm_run = random.Random(self.seed_algorithm_run)
        print(f"Algorithm run seed: {self.seed_algorithm_run}")

        # compute dfg number of nodes
        self.NNODES = len(list(self.dfg))
 
        # build NODE_SCHEDULE_T
        for t in self.schedule:
            for n in self.schedule[t]:
                self.NODE_SCHEDULE_T[n] = t


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
    

    @staticmethod
    def build_pe_nodes(node_pe: dict[int, int]) -> dict[int, list[int]]:
        pe_nodes: dict[int, list[int]] = {}

        for n in node_pe:
            pe = node_pe[n]
            if pe not in pe_nodes:
                pe_nodes[pe] = []
            pe_nodes[pe].append(n)

        return pe_nodes
    

    def pe_distance(self, pe1: int, pe2: int) -> int:
        """
        Computes manhattan distance of the shortest path from pe1 to pe2 considering CGRA wrap around

        Args:
            pe1 (int): First PE
            pe2 (int): Second PE

        Returns:
            int: Distance in number of edges between pe1 and pe2 considering wrap-around
        """
        # row pe1
        r1 = pe1 // self.size_y
        # column pe1
        c1 = pe1 % self.size_y

        # row pe2
        r2 = pe2 // self.size_y
        # column pe2
        c2 = pe2 % self.size_y

        row_dist = abs(r1 - r2)
        col_dist = abs(c1 - c2)

        # as it is a torus wrap around can be considered as subtracting the distance of inside
        # edges from the total length: think of it self.size_x is the number of edges from
        # one node following the column, wrapping around and coming back, then if the rows distance
        # of one node in the board to the other node in the board without wrap around is row_dist the
        # number of edges you need to walk from that node to the other wrapping around is self.size_x - row_dist
        return min(row_dist, self.size_x - row_dist) + min(col_dist, self.size_y - col_dist)

    
    # Implementation Specific (Strategy)
    def neighbour_sol_generator(self):
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        The implementation differs from strategy to strategy

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """
        raise NotImplementedError

    def undo_neighbour_sol_generator(self):
        """
        Undo the last Neighbour solution generator action.
        The function uses the last geenrate neighbor solution to apply the inverse change applyed by `neighbour_sol_generator`

        The implementation differs from strategy to strategy.
        The neighbour solution should return to be equal to the current best startegy
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

    # Generation of an initial solution
    def initial_sol_generator(self):
        """
        Initial solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.

        This implementation generates a random initial solution
        """
        raise NotImplementedError


    # Semi-Hybrid: Some strategies may add extra computation to the cost function
    def cost_space_solution(self, curr_node_pe: dict[int, int], p: str | None = None, silent: bool = False) -> int:
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
            if silent:
                if source not in curr_node_pe:
                    continue
                if destination not in curr_node_pe:
                    continue
            else:
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
    
    # Events like function
    # additional setup post construction of constant
    def __post_costant_initialization__(self):
        pass

    # routine that children may override and can be called once temperature is reset to trigger
    # specific mutations
    def __temperature_reset__(self):
        pass

    # routine that children may override and can be called once temperature is reset to trigger
    # extra plots
    def __extra_plots__(self, extra_plot_path: str | None):
        pass

    # Overall Temperature and Simulated Annealing search is shared
    def simulatedAnnealingSearch(self) -> tuple[dict[int, int],  dict[int, list[int]], float]:
        # prepare id: it is also a check that all needed classes have been added
        if not self.START_ID:
            raise AssertionError("START is not set")
        if not self.ROUTINE_ID:
            raise AssertionError("ROUTINE is not set")
        if not self.STRATEGY_ID:
            raise AssertionError("STRATEGY is not set")
        
        # last constant that needs constructors to be succesfull
        self.ID = f"{self.START_ID}_{self.ROUTINE_ID}_{self.STRATEGY_ID}"

        print(f"Running: {self.ID}")

        # prepare temperature
        if not self.START_TEMPERATURE:
            raise AssertionError("Start temperature must be set before calling simulatedAnnealingSearch")
        self.temperature = self.START_TEMPERATURE

        # post CONSTANTs initialization
        self.__post_costant_initialization__()

        # we set benchmarker and recorder if present
        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.ID)

        print()
        print("*** START SA ROUTINE ***\n")
        self.start_time = time.process_time()

        # Generate starting random solution and evaluate it
        self.initial_sol_generator()
        # first solution: deep copy
        self.curr_node_pe = self.node_pe.copy()

        # compute solution cost
        self.sol_cost = self.cost_space_solution(self.node_pe)
        self.start_configuration_cost = self.sol_cost

        # init MA
        self.cost_sma_fast = float(self.sol_cost)
        self.cost_sma_slow = float(self.sol_cost)

        # we warm data collectors
        self.costs.append(self.sol_cost)
        self.temperatures.append(self.temperature)
        self.probabilities.append(1)
        
        if isinstance(self.costs_sma_fast, list):
            self.costs_sma_fast.append(self.cost_sma_fast)
        if isinstance(self.costs_sma_slow, list):
            self.costs_sma_slow.append(self.cost_sma_slow)

        self.iterations = 1

        # Main loop
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

        # TODO adjust
        self.__extra_plots__(plot_files_path)

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

                average_neighbor_sol_time_item =(self.cumulative_neighbor_sol_time_item / self.total_items_iterations) if self.total_items_iterations > 0 else 0,
                average_cost_space_sol_time_item =(self.cumulative_cost_space_sol_time_item / self.total_items_iterations)  if self.total_items_iterations > 0 else 0,
                average_sol_check_items_routine_time =(self.cumulative_sol_check_items_routine_time / self.iterations) if self.iterations > 0 else 0,
                average_temp_routine_time =(self.cumulative_temp_routine_time / self.iterations) if self.iterations > 0 else 0,
                average_running_time =(self.cumulative_running_time / self.iterations) if self.iterations > 0 else 0
            )

        if self.RECORDER:
            self.RECORDER.record_run(
                costs=self.costs,
                temperatures=self.temperatures,
                probabilities=self.probabilities,
                
                # optionally pass in
                **({"costs_sma_fast": self.costs_sma_fast,
                    "costs_sma_slow": self.costs_sma_slow}
                    if isinstance(self.costs_sma_fast, list) and isinstance(self.costs_sma_slow, list)
                    else {})
            )

        print()
        return (self.node_pe, self.pe_nodes, total_time)
