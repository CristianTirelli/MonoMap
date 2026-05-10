import math
import time
from dataclasses import dataclass, field

from utils.mappings_json import mappings_to_json_file
from plots import save_plot_mappings_cgra
from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch


@dataclass
class CoolingResetToProbabilityDynamicBestCostSmaRestart(SimulatedAnnealingSpaceSearch):
    """
    As soon as we plateau multiple times at the same or at a worst plateau we start from the beginning.

    We remove all operations from the board and we place each operation one be one, once we reached a layout without overlaps and all nodes placed we resume with the usual simulated annealing.
    """

    overlapping_nodes: list[int] = field(default_factory=list, init=False)

    PLATEAU_TIMES: int = field(default=3, init=False)
    
    RESET_PROBABILITY: float = field(default=0.45, init=False)

    FREEZING_TEMPERATRE: float = field(default=0.001, init=False)

    def nodes_pes_overlap_count(self, curr_node_pe: dict[int, int], build_overlapping_nodes: bool = False) -> int:
        overlaps = 0
        if build_overlapping_nodes:
            self.overlapping_nodes.clear()

        for n in curr_node_pe:
            t = self.NODE_SCHEDULE_T[n]

            for nneighbor in self.schedule[t]:
                if nneighbor != n:
                    if nneighbor in curr_node_pe and curr_node_pe[nneighbor] == curr_node_pe[n]:
                        overlaps += 1

                        # update list: costly, for now ok
                        if build_overlapping_nodes and nneighbor not in self.overlapping_nodes:
                            self.overlapping_nodes.append(nneighbor)
                        if build_overlapping_nodes and n not in self.overlapping_nodes:
                            self.overlapping_nodes.append(n)
        return int(overlaps / 2)

    def __post_init__(self):
        super().__post_init__()
        self.overlapping_nodes = []

        NNODES_DFG = len(list(self.dfg))

        self.START_TEMPERATURE = NNODES_DFG * self.START_TEMPERATURE_COEFF
        self.ITEMS_PER_TEMPERATRE = NNODES_DFG * 10

        self.ROUTINE_ID = f"COOLING-RESET-TO-{str(self.RESET_PROBABILITY).replace(".", "-")}-DYNAMIC-BEST-COST-START-T-COEFF-{self.START_TEMPERATURE_COEFF}-RESTART"

    # Hybrid: Overall Temperature and Simulated Annealing search is shared
    def temperature_routine(self):

        plateau_counter: int = 0
        last_plateau_cost: int = 0

        while self.sol_cost != 0 and self.iterations < self.MAX_ITERATIONS:
            while_running_time = time.process_time()

            running_since = time.process_time() - self.start_time
            if self.TIME_OUT < running_since:
                print("TIMED OUT")
                break

            if plateau_counter == self.PLATEAU_TIMES:
                print("Restarting SA search with new solution: Computing solution")
                # TODO think about it: true?
                # can be also have richer search: we can keep track of the starting tree of operations
                # then if the starting tree is the same or looks the same we reject it and recompute
                # a new starting solution.

                # TODO think about it: true?
                # can also have precise restart: we can reach here not by plateau but by tabu list that sees all operations
                # have been tried but did not work properly

                # we compute a completely new starting solution using temperature
                # map all oprations to a random pe

                # we reach this point at a freezing temperature and we want to be cold
                # as we aim at computing an already good solution

                self.temperature = self.FREEZING_TEMPERATRE
                size = self.size_x * self.size_y

                # a random initial pe to position all operations
                initial_pe = self.randgen_algorithm_run.randint(0, size - 1)
                for n in self.node_pe:
                    self.node_pe[n] = initial_pe

                # still we dont want to have a too great temperature here
                # as otherwise the starting solution will be completely divergent
                # how to normalize the temperature in a good way? -> we reach this point at freezing T

                # interesting variation from starts centered as we do not select from the overlapping list
                # but from the general list
                c: int = self.cost_space_solution(self.node_pe)
                coverlap: int = self.nodes_pes_overlap_count(self.node_pe, build_overlapping_nodes=True)
                while 0 < coverlap:
                    sitems: int = 0

                    running_since = time.process_time() - self.start_time
                    if self.TIME_OUT < running_since:
                        print("TIMED OUT")
                        break

                    # also to consider: it is easy to find a solution that gives a zero overlappig
                    # cost but still gives a very high solution cost, should we consider?
                    while 0 < coverlap and sitems < self.ITEMS_PER_TEMPERATRE:
                        # select random node
                        node_to_move = self.randgen_algorithm_run.choice(self.overlapping_nodes)
                        # select pe to move to
                        pe_to_move = self.randgen_algorithm_run.randint(0, size - 1)

                        # perturb solution
                        old_node_to_move_pe = self.node_pe[node_to_move]
                        self.node_pe[node_to_move] = pe_to_move

                        # compute overlaps
                        curr_coverlap = self.nodes_pes_overlap_count(self.node_pe, build_overlapping_nodes=False)
                        # compute cost
                        curr_c = self.cost_space_solution(self.node_pe)

                        # delta overlap: minimization
                        dcoverlap = curr_coverlap - coverlap
                        # delta cost
                        dt = curr_c - c

                        if dcoverlap < 0 and dt < 0:
                            # accept right away
                            c = curr_c
                            coverlap = curr_coverlap

                            # self.node_pe is maintained
                            self.nodes_pes_overlap_count(self.node_pe, build_overlapping_nodes=True)
                        else:
                            # apply sa boltzman
                            dE = (abs(dcoverlap) ** 2 + abs(dt)) / self.temperature
                            P = math.exp(- dE)

                            rnd = self.randgen_algorithm_run.random()
                            if rnd < P:
                                # accepted
                                c = curr_c
                                coverlap = curr_coverlap

                                # self.node_pe is maintained
                                self.nodes_pes_overlap_count(self.node_pe, build_overlapping_nodes=True)
                            else:
                                # undo neighbor decision
                                self.node_pe[node_to_move] = old_node_to_move_pe
                        sitems += 1

                    print(f"{int(running_since):4d}s C: {c:6d} COVR: {coverlap:4d} T: {self.temperature:4.4f}", end='\r')
                    # we have seen all items: remember we are freezing, we reheat a little
                    self.temperature *= 1.01

                # we are done and we found a new starting solution
                # rebuild pe_nodes from node_pe
                self.pe_nodes.clear()
                for n, pe in self.node_pe.items():
                    if pe not in self.pe_nodes:
                        self.pe_nodes[pe] = []
                    self.pe_nodes[pe].append(n)

                # reset T: we start from scratch T
                self.temperature = self.START_TEMPERATURE
                self.sol_cost = c

                # clean plateau costs
                last_plateau_cost = 0
                plateau_counter = 0

                print(f"Restarting SA search with solution cost: {self.sol_cost} from temperature: {self.temperature}")

                # TODO comment when running benchmarks
                # id_sol = f"cooling_reset_to_probability_restart_at_{int(running_since)}s"
                # save_plot_mappings_cgra(self.node_pe, self.schedule, self.size_x, self.size_y, id=id_sol)
                # mappings_to_json_file(self.node_pe, self.pe_nodes, self.sol_cost, path=id_sol)
                continue

            average_p: float = 0
            times_we_generated_p: int = 0

            before_items_sol_cost: int = self.sol_cost

            items: int = 0
            acceptance: int = 0
            before_sol_check_items_routine_time = time.process_time()
            while self.sol_cost != 0 and items < self.ITEMS_PER_TEMPERATRE:
                before_neighbor_sol_time_item = time.process_time()
                self.neighbour_sol_generator()
                self.cumulative_neighbor_sol_time_item += (time.process_time() - before_neighbor_sol_time_item)

                before_cost_space_sol_time_item = time.process_time()
                c = self.cost_space_solution(self.curr_node_pe)
                self.cumulative_cost_space_sol_time_item += (time.process_time() - before_cost_space_sol_time_item)

                dt = c - self.sol_cost
                if dt < 0:
                    self.node_pe = self.curr_node_pe.copy()
                    self.pe_nodes = SimulatedAnnealingSpaceSearch.build_pe_nodes(self.node_pe)
                    self.sol_cost = c

                    acceptance += 1
                else:
                    dE = dt / self.temperature
                    P = math.exp(- dE)

                    average_p += P
                    times_we_generated_p += 1

                    rnd = self.randgen_algorithm_run.random()

                    if rnd < P:
                        self.node_pe = self.curr_node_pe.copy()
                        self.pe_nodes = SimulatedAnnealingSpaceSearch.build_pe_nodes(self.node_pe)
                        self.sol_cost = c

                        acceptance += 1
                    else:
                        self.undo_neighbour_sol_generator()

                items += 1
                self.total_items_iterations += 1
            self.cumulative_sol_check_items_routine_time += (time.process_time() - before_sol_check_items_routine_time)
            
            before_temperature_routine_time = time.process_time()
            # New average = old average * (n-1)/n + new value /n
            self.cost_sma_fast = self.cost_sma_fast * ((self.SMA_FAST_ITEMS - 1) / self.SMA_FAST_ITEMS) + self.sol_cost / self.SMA_FAST_ITEMS
            self.cost_sma_slow  = self.cost_sma_slow * ((self.SMA_SLOW_ITEMS - 1) / self.SMA_SLOW_ITEMS) + self.sol_cost / self.SMA_SLOW_ITEMS

            # data
            self.costs.append(self.sol_cost)
            self.temperatures.append(self.temperature)

            self.probabilities.append(
                average_p / times_we_generated_p if times_we_generated_p > 0 and before_items_sol_cost != self.sol_cost
                    else (self.probabilities[-1] if len(self.probabilities) > 1 else 1 ** -16))

            self.costs_sma_fast.append(self.cost_sma_fast)
            self.costs_sma_slow.append(self.cost_sma_slow)

            # acceptance rate
            acceptance_rate = acceptance / self.ITEMS_PER_TEMPERATRE

            # t update
            self.temperature *= 0.9

            # reheating
            if abs(self.cost_sma_fast - self.cost_sma_slow) / max(self.cost_sma_fast, self.cost_sma_slow) < self.SMA_REHEATING_THRESHOLD_PERCENTAGE and acceptance_rate < self.ACCEPTANCE_RATE_REHEATING_THRESHOLD_PERCENTAGE:
                if last_plateau_cost == 0:
                    last_plateau_cost = self.sol_cost
                else:
                    if last_plateau_cost <= self.sol_cost:
                        plateau_counter += 1

                # here we probably delay the spike if we will recompute the start: it will be the spike manager
                if plateau_counter < self.PLATEAU_TIMES:
                    # compute the delta c = self.sol_cost
                    self.temperature = self.sol_cost / math.log(1 / self.RESET_PROBABILITY)

                last_plateau_cost = self.sol_cost


            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:4d} LPCT: {last_plateau_cost:4d} PCR: {plateau_counter:2d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  T: {self.temperature:4.4f}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)

            # total while running time
            self.cumulative_running_time += (time.process_time() - while_running_time)
