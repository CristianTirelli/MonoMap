import math
import time
from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

# 1) Start with random configuration. At each iteration  I take a node and choose if move it in another PE or not (and swap with other nodes too)
# 2) I take a temperature T = 1000 and at each teration I decrease: T = 0.9*T
# 3) At each iteration I do more movements (double loop): number_of_nodes_dfg*100.
# 4) I change solution if the choiced solution improves my function cost or if P < F. Where P is a random number [0, 1].
# F defined as: F = e^(- dt / T ).
# dt defined as: dt = cost_after - cost_before

# Pseudo code:

# MAX_NUM_IT = 1000
# T = CGRA_SIZE*100
# config = randomPlacement(dfg)
# cost = computeCost(config)
# For i in MAX_NUM_IT{
#   For j in num_dfg_nodes*100{
#     r = random() //numero random tra 0 e 1
#     current_config = moveRandomNode(config)
#     cost_before = cost
#     cost_after = computeCost( current_config)
#     dt = cost_after - cost_before
#     F = e^(-dt/T)
#     if dt < 0 or r < F{
#       config = current_config // make move
#     }
#  }
#  T = T*0.9
# }

@dataclass
class CoolingResetSma(SimulatedAnnealingSpaceSearch):
    TEMPERATURE_STRATEGY_ID: str = field(default="COOLING-RESET_", init=False)

    def __post_init__(self):
        super().__post_init__()
    
        self.START_TEMPERATURE = self.size_x * self.size_y * 100
        self.ITEMS_PER_TEMPERATRE = len(list(self.dfg)) * 10

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
            self.probabilities.append(average_p / times_we_generated_p if times_we_generated_p > 0 else None)
            self.costs_sma_fast.append(self.cost_sma_fast)
            self.costs_sma_slow.append(self.cost_sma_slow)

            # acceptance rate
            acceptance_rate = acceptance / self.ITEMS_PER_TEMPERATRE

            # t update
            self.temperature *= 0.9

            # reheating
            if self.cost_sma_fast < self.cost_sma_slow and self.cost_sma_slow < self.cost_sma_fast + self.EPSILON and acceptance_rate < 0.1:
                self.temperature = self.START_TEMPERATURE
                self.__temperature_reset__()

            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:6d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  T: {self.temperature:4.4f}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)

            # total while running time
            self.cumulative_running_time += (time.process_time() - while_running_time)
