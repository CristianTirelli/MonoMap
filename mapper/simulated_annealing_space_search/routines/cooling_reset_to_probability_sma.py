import math
import time
from dataclasses import dataclass, field

import numpy as np

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

@dataclass
class FixedCoolingResetToProbabilitySma(SimulatedAnnealingSpaceSearch):
    RESET_PROBABILITY: float = field(default=0.6, init=False)

    PROBABILITIES_TABLE: list[tuple[int, int]] = field(default_factory=list, init=False)
    PROBABILITIES_STEPS: int = field(default=0, init=False)

    t_index: int = field(default=0, init=False)

    MAX_PLATEAU_TIMES: int = field(default=100, init=False)

    def __post_init__(self):
        super().__post_init__()

        # we remove SMAs from plots
        self.costs_sma_fast = None
        self.costs_sma_slow = None
    
        self.START_TEMPERATURE = 1.0
        self.ITEMS_PER_TEMPERATRE = len(list(self.dfg)) * 10

        # we make non-linear steps, which are less at higher ts and more at lower ts
        first = np.linspace(1, 0.001, 200, endpoint=False)
        fourth = np.linspace(0.001, 0.00001, 2000, endpoint=False)
        steps = np.concatenate([first, fourth])

        MAX_CAP = 1_000

        self.PROBABILITIES_STEPS = len(steps)
        for stp in steps:
            # we make a list with tuples that hold probabilities and items to be seen within such probability
            # we apply the heuristic that for lower probabilities we want to see more items
            # we apply cubic proportion for now:
            coeff_items = (self.START_TEMPERATURE - stp)
            additional_items = int(
                ((coeff_items ** 2 if coeff_items > 1 else coeff_items) * (self.ITEMS_PER_TEMPERATRE))
            )
            total_items =  self.ITEMS_PER_TEMPERATRE + additional_items
            self.PROBABILITIES_TABLE.append((stp, total_items if total_items < MAX_CAP else MAX_CAP))

        self.ITEMS_PER_TEMPERATRE = self.PROBABILITIES_TABLE[-1][1]
        self.ROUTINE_ID: str = f"FIXED-P-RESET-TO-{str(self.RESET_PROBABILITY).replace(".", "-")}"

    # Hybrid: Overall Temperature and Simulated Annealing search is shared
    def temperature_routine(self):
        plateau_times: int = 0

        while self.sol_cost != 0 and self.iterations < self.MAX_ITERATIONS:
            while_running_time = time.process_time()

            running_since = time.process_time() - self.start_time
            if self.TIME_OUT < running_since:
                print("TIMED OUT")
                break

            dynamic_items_per_temperature: int = self.PROBABILITIES_TABLE[self.t_index][1]
            items: int = 0
            acceptance: int = 0
            before_sol_check_items_routine_time = time.process_time()
            while self.sol_cost != 0 and items < dynamic_items_per_temperature:
                before_neighbor_sol_time_item = time.process_time()
                self.neighbour_sol_generator()
                self.cumulative_neighbor_sol_time_item += (time.process_time() - before_neighbor_sol_time_item)

                before_cost_space_sol_time_item = time.process_time()
                c = self.cost_space_solution(self.curr_node_pe)
                self.cumulative_cost_space_sol_time_item += (time.process_time() - before_cost_space_sol_time_item)

                dt = c - self.sol_cost
                if dt <= 0:
                    self.node_pe = self.curr_node_pe.copy()
                    self.pe_nodes = SimulatedAnnealingSpaceSearch.build_pe_nodes(self.node_pe)
                    self.sol_cost = c

                    acceptance += 1
                else:
                    P = self.temperature
                    P /= dt ** 1.125

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

            # data
            self.costs.append(self.sol_cost)
            self.temperatures.append(self.temperature)
            self.probabilities.append(self.temperature)

            # acceptance rate
            acceptance_rate = acceptance / self.ITEMS_PER_TEMPERATRE

            # t update by indexing
            if self.t_index + 1 == self.PROBABILITIES_STEPS:
                # reheating
                # we reach 0 we reheat based on plateau times
                plateau_times += 1
                if plateau_times >= self.MAX_PLATEAU_TIMES:
                    # find nearest like t from available steps
                    seeked_p = self.RESET_PROBABILITY
                    best_fit_delta = 1.01
                    best_fit_idx = 0
                    for i in range(len(self.PROBABILITIES_TABLE)):
                        fit_delta = abs(self.PROBABILITIES_TABLE[i][0] - seeked_p)
                        if fit_delta < best_fit_delta:
                            best_fit_delta = fit_delta
                            best_fit_idx = i

                    plateau_times = 0
                    self.t_index = best_fit_idx
                    self.temperature = self.PROBABILITIES_TABLE[best_fit_idx][0]
            else:
                # else we step to next temperature in the list
                self.t_index += 1
                self.temperature = self.PROBABILITIES_TABLE[self.t_index][0]


            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:6d} PLATEAU_T: {plateau_times} T: {self.temperature:4.4f} items: {dynamic_items_per_temperature}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)

            # total while running time
            self.cumulative_running_time += (time.process_time() - while_running_time)


@dataclass
class CoolingResetToProbabilityDynamicLearnedSma(SimulatedAnnealingSpaceSearch):
    RESET_PROBABILITY: float = field(default=0.6, init=False)

    # holds the map: probability -> temperature
    probability_to_temperature: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        super().__post_init__()
    
        NNODES_DFG = len(list(self.dfg))

        self.START_TEMPERATURE = NNODES_DFG * self.START_TEMPERATURE_COEFF
        self.ITEMS_PER_TEMPERATRE = len(list(self.dfg)) * 10

        self.ROUTINE_ID: str = f"COOLING-RESET-TO-{str(self.RESET_PROBABILITY).replace(".", "-")}-DYNAMIC-LEARNED-START-T-COEFF-{self.START_TEMPERATURE_COEFF}"

    # Hybrid: Overall Temperature and Simulated Annealing search is shared
    def temperature_routine(self):
        while self.sol_cost != 0 and self.iterations < self.MAX_ITERATIONS:
            while_running_time = time.process_time()

            running_since = time.process_time() - self.start_time
            if self.TIME_OUT < running_since:
                print("TIMED OUT")
                break

            average_p: float = 0
            times_we_generated_p: int = 0

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
                    self.node_pe = self.curr_node_pe
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
                        self.node_pe = self.curr_node_pe
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
            self.probabilities.append(average_p / times_we_generated_p if times_we_generated_p > 0 else None)
            self.costs_sma_fast.append(self.cost_sma_fast)
            self.costs_sma_slow.append(self.cost_sma_slow)

            # add probability and its T to the array of learned temperatures
            if times_we_generated_p > 0:
                p = int((average_p / times_we_generated_p) * 100)

                if p in self.probability_to_temperature:
                    self.probability_to_temperature[p] = (self.probability_to_temperature[p] + self.temperature) / 2
                else: 
                    self.probability_to_temperature[p] = self.temperature

            # acceptance rate
            acceptance_rate = acceptance / self.ITEMS_PER_TEMPERATRE

            # t update
            self.temperature *= 0.9

            # reheating
            if abs(self.cost_sma_fast - self.cost_sma_slow) / max(self.cost_sma_fast, self.cost_sma_slow) < self.SMA_REHEATING_THRESHOLD_PERCENTAGE and acceptance_rate < self.ACCEPTANCE_RATE_REHEATING_THRESHOLD_PERCENTAGE:
                seek_p = int(self.RESET_PROBABILITY * 100)
                if seek_p in self.probability_to_temperature:
                    next_t = self.probability_to_temperature[seek_p]
                    print(f"reset by learned t to: {next_t} aiming for and got: {seek_p}")
                    self.temperature = next_t
                else:
                    # seek near line tempeatures, with a change of +-5, take first that appears
                    found = False
                    for k, v in self.probability_to_temperature.items():
                        change_from_seek_p = abs(k - seek_p)
                        if change_from_seek_p <= 5:
                            print(f"reset by near t with +- change of: {change_from_seek_p} to: {v} aiming for: {seek_p} we got: {k}")
                            self.temperature = v
                            found = True
                            break

                    if not found:
                        # we have no knowledge: default to computed
                        self.temperature = self.sol_cost / math.log(1 / self.RESET_PROBABILITY)
                        print("reset by computed t to: {self.temperature}")

            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:6d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  T: {self.temperature:4.4f}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)

            # total while running time
            self.cumulative_running_time += (time.process_time() - while_running_time)


@dataclass
class CoolingResetToProbabilityDynamicBestCostSma(SimulatedAnnealingSpaceSearch):
    RESET_PROBABILITY: float = field(default=0.45, init=False)

    # here for benchmarks rn
    def __post_init__(self):
        super().__post_init__()
    
        NNODES_DFG = len(list(self.dfg))

        self.START_TEMPERATURE = NNODES_DFG * self.START_TEMPERATURE_COEFF
        self.ITEMS_PER_TEMPERATRE = NNODES_DFG * 10

        self.ROUTINE_ID = f"COOLING-RESET-TO-{str(self.RESET_PROBABILITY).replace(".", "-")}-DYNAMIC-BEST-COST-START-T-COEFF-{self.START_TEMPERATURE_COEFF}"

    # Hybrid: Overall Temperature and Simulated Annealing search is shared
    def temperature_routine(self):
        while self.sol_cost != 0 and self.iterations < self.MAX_ITERATIONS:
            while_running_time = time.process_time()

            running_since = time.process_time() - self.start_time
            if self.TIME_OUT < running_since:
                print("TIMED OUT")
                break

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
            if abs(self.cost_sma_fast - self.cost_sma_slow) / min(self.cost_sma_fast, self.cost_sma_slow) < self.SMA_REHEATING_THRESHOLD_PERCENTAGE and acceptance_rate < self.ACCEPTANCE_RATE_REHEATING_THRESHOLD_PERCENTAGE:
                # compute the delta c = self.sol_cost
                self.temperature = self.sol_cost / math.log(1 / self.RESET_PROBABILITY)

            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:6d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  T: {self.temperature:4.4f}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)

            # total while running time
            self.cumulative_running_time += (time.process_time() - while_running_time)
