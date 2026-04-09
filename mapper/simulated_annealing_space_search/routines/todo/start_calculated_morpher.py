import math
import random
import time
from dataclasses import dataclass, field

from networkx import Graph, shortest_path_length

from benchmark import Benchmark
from plots import saveFigTemperature

# TODO unconnected old implementation class

# ALGORITHM:    We have a max cap of iterations 1_000_000 and a temperature that is detached from the main loop,
#               until we reach a solution, we complete all iterations or we timeout we continue the search.

# START:        We compute the first solution by an heuristic to reduce the starting cost as much as possible
#               we place operations on vicinity PEs of dependent PEs, if there are overlapping (two operations
#               scheduled) on the same PE, we move one of them away in a neighbouring available PE so that the
#               cost is minimized

# TEMPERATURE:  Temperature moves down following Morpher cooling schedule, and SMA: 201 and 5, if 201 reaches 5
#               we warm up following a schedule: += 0.001 / math.log(warming_up_step + math.e) then if we find a
#               better solution or we reach a cap of 1000 times the freezing temperature we stop warming up and
#               continue cooling

# COST:         The cost function simply calculates the Manhattan distance of nodes that do not resespect dependencies

@dataclass
class StartCalculatedMorpher:
    TEMPERATURE_STRATEGY_ID: str = field(default="MORPHER-SMA-BEST-AND-MAX-CAP-COMPUTED-START_", init=False)

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
    TIME_OUT: int = field(default=60 * 30, init=False)
    # TIME_OUT: int = field(default=4000, init=False)

    # temperature
    START_TEMPERATURE: int = field(default=100, init=False)
    FREEZING_TEMPERATURE: float = field(default=0.01, init=False)
    TEMPERATURE_REFUEL_LIMIT: float = field(default=10, init=False)

    ITEMS_PER_TEMPERATRE: int = field(default=5, init=False)

    # SMA
    SMA_SLOW_ITEMS: float = field(default=201, init=False)
    SMA_FAST_ITEMS: float = field(default=5, init=False)

    EPSILON: float = field(default=0.001, init=False)

    # Shared functions
    def start_sol_generator(self):
        """
        Random solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.
        """
        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # invece di andare a caso guardare il neghbour con la funzione costo e
        # scegliere la posizione che minimizza il costo

        # Very naive: we place all pes on the same tile for each schedule
        middle = size // 2

        for t in self.schedule:
            for n in self.schedule[t]:
                curr_node_pe[n] = middle

        def is_node_overlapping(n: int, curr_node_pe: dict[int, int], neighbours: list[int]) -> bool:
            for nei in neighbours:
                if n != nei and curr_node_pe[n] == curr_node_pe[nei]:
                    return True
            return False

        # at random for each schedule we chose one of the adjacent PEs we continue until
        # all operations are on a single pe
        for t in self.schedule:
            for n in self.schedule[t]:
                while is_node_overlapping(n, curr_node_pe, self.schedule[t]):
                    cnpe = curr_node_pe[n]
                    adjacent_pes = [pe for pe in [cnpe + 1, cnpe - 1, cnpe + self.size_x, cnpe - self.size_x] if 0 <= pe < size]

                    new_pe = random.choice(adjacent_pes)
                    curr_node_pe[n] = new_pe

        # build curr_pe_nodes
        for n, pe in curr_node_pe.items():
            if pe not in curr_pe_nodes:
                curr_pe_nodes[pe] = []
            curr_pe_nodes[pe].append(n)

        # apply to main variables
        self.node_pe = curr_node_pe
        self.pe_nodes = curr_pe_nodes

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
        print("*** START SA ROUTINE ***\n")
        start = time.process_time()

        self.temperature = self.START_TEMPERATURE

        # Generate starting random solution and evaluate it
        self.start_sol_generator()
        self.sol_cost = self.cost_space_solution(self.node_pe)

        # Search Data
        costs = []
        temperatures = []
        costs_sma_slow = []
        costs_sma_fast = []

        # Search local variables
        self.cost_sma_fast: float = float(self.sol_cost)
        self.cost_sma_slow: float = float(self.sol_cost)

        acceptance = 0
        items = 0

        warming_up = False
        warming_up_step = 1

        iterations = 0
        while self.sol_cost != 0 and iterations < self.MAX_ITERATIONS:
            running_time = time.process_time() - start

            if self.TIME_OUT < running_time:
                print("TIMED OUT")
                break

            # add best pe nodes here
            curr_node_pe, curr_pe_nodes = self.neighbour_sol_generator()
            c = self.cost_space_solution(curr_node_pe)

            if c < self.sol_cost:
                self.node_pe = curr_node_pe
                self.pe_nodes = curr_pe_nodes
                self.sol_cost = c

                acceptance += 1
            else:
                # apply boltzman
                delta_E = c - self.sol_cost
                P = math.exp(- delta_E / self.temperature)
                rnd = random.random()

                if rnd < P:
                    self.node_pe = curr_node_pe
                    self.pe_nodes = curr_pe_nodes
                    self.sol_cost = c

                    acceptance += 1

            if items >= self.ITEMS_PER_TEMPERATRE - 1:
                # New average = old average * (n-1)/n + new value /n
                self.cost_sma_fast = self.cost_sma_fast * ((self.SMA_FAST_ITEMS - 1) / self.SMA_FAST_ITEMS) + self.sol_cost / self.SMA_FAST_ITEMS
                self.cost_sma_slow  = self.cost_sma_slow * ((self.SMA_SLOW_ITEMS - 1) / self.SMA_SLOW_ITEMS) + self.sol_cost / self.SMA_SLOW_ITEMS

                # track what we keep from these items within the single temperature step
                costs.append(self.sol_cost)
                temperatures.append(self.temperature)
                costs_sma_fast.append(self.cost_sma_fast)
                costs_sma_slow.append(self.cost_sma_slow)

                acceptance_rate = acceptance / self.ITEMS_PER_TEMPERATRE

                print(f"Running for: {int(running_time):4d}s  Warming up: {str(warming_up)}  Acceptance rate: {acceptance_rate:2.2f}  Cost: {self.sol_cost:6d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  Temperature: {self.temperature:4.4f}", end='\r')

                items = 0
                acceptance = 0

                if warming_up:
                    # when should i stop warming up? when can i be mostly sure that we escaped a possible stuck local minimum?
                    # if we find a result that is less costly than the current best result?

                    # t increase
                    self.temperature += 0.001 / math.log(warming_up_step + math.e)

                    # based on temperature cap
                    if self.temperature + self.EPSILON >= self.TEMPERATURE_REFUEL_LIMIT:
                        warming_up = False

                    # based on better solutions found
                    if self.cost_sma_fast + self.EPSILON < self.cost_sma_slow:
                        warming_up = False
                    warming_up_step += 1
                else:
                    if self.cost_sma_fast < self.cost_sma_slow and self.cost_sma_slow < self.cost_sma_fast + self.EPSILON and acceptance_rate < 0.1:
                        warming_up = True
                        warming_up_step = 1

                    if self.temperature > self.FREEZING_TEMPERATURE:
                        self.temperature = self.updateTemperature(self.temperature, acceptance_rate)
                iterations += 1
            else:
                items += 1
        end = time.process_time()
        total_time = end - start

        print("\n*** END SA ROUTINE ***\n")
        print(f"End solution cost: {self.sol_cost}")

        # Plot search data
        if self.BENCHMARK:
            file_path: str = ""
            try:
                file_path = saveFigTemperature(
                    iterations,
                    temperatures,
                    costs,
                    self.BENCHMARK.get_directory_path_str(),
                    self.BENCHMARK.sa_algorithm_type,
                    self.BENCHMARK.id,
                    costs_sma_slow=costs_sma_slow,
                    costs_sma_fast=costs_sma_fast
                )
            except:
                pass

            self.BENCHMARK.save_results(total_time, self.sol_cost, iterations, file_path)
        else:
            saveFigTemperature(iterations, temperatures, costs, costs_sma_slow, costs_sma_fast)
        
        print("\n\n\n")
        return (self.node_pe, self.pe_nodes, total_time)
