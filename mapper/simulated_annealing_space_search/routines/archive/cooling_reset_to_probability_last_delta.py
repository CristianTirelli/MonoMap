
from dataclasses import dataclass, field
import math
import time

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch


@dataclass
class CoolingResetToProbabilityDynamicLastDeltaSma(SimulatedAnnealingSpaceSearch):
    # fixed to 0.5 for T tests
    RESET_PROBABILITY: float = field(default=0.45, init=False)

    def __post_init__(self):
        super().__post_init__()
    
        NNODES_DFG = len(list(self.dfg))

        self.START_TEMPERATURE = NNODES_DFG * self.START_TEMPERATURE_COEFF
        self.ITEMS_PER_TEMPERATRE = NNODES_DFG * 10

        self.ROUTINE_ID: str = f"COOLING-RESET-TO-{str(self.RESET_PROBABILITY).replace(".", "-")}-DYNAMIC-LAST-DELTA-START-T-COEFF-{self.START_TEMPERATURE_COEFF}"

    # Hybrid: Overall Temperature and Simulated Annealing search is shared
    def temperature_routine(self):
        last_delta_cost_improvment: int = self.sol_cost

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
                curr_node_pe, curr_pe_nodes = self.neighbour_sol_generator()
                self.cumulative_neighbor_sol_time_item += (time.process_time() - before_neighbor_sol_time_item)

                before_cost_space_sol_time_item = time.process_time()
                c = self.cost_space_solution(curr_node_pe)
                self.cumulative_cost_space_sol_time_item += (time.process_time() - before_cost_space_sol_time_item)

                dt = c - self.sol_cost
                if dt < 0:
                    self.node_pe = curr_node_pe
                    self.pe_nodes = curr_pe_nodes
                    self.sol_cost = c

                    acceptance += 1
                else:
                    dE = dt / self.temperature
                    P = math.exp(- dE)

                    average_p += P
                    times_we_generated_p += 1

                    rnd = self.randgen_algorithm_run.random()

                    if rnd < P:
                        self.node_pe = curr_node_pe
                        self.pe_nodes = curr_pe_nodes
                        self.sol_cost = c

                        acceptance += 1

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

            # improvment
            if self.sol_cost < before_items_sol_cost:
                last_delta_cost_improvment = before_items_sol_cost - self.sol_cost

            # reheating
            if self.cost_sma_fast < self.cost_sma_slow and self.cost_sma_slow < self.cost_sma_fast + self.EPSILON and acceptance_rate < 0.1:
                # compute the delta c = self.sol_cost
                self.temperature = last_delta_cost_improvment / math.log(1 / self.RESET_PROBABILITY)

            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:6d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  T: {self.temperature:4.4f}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)

            # total while running time
            self.cumulative_running_time += (time.process_time() - while_running_time)
