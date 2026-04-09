import math
import time
from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

# TODO: Simulated annealing superclass that leaves out temperature routina and solution routine implementation
#       then you have set of temperature routie stategies that inherit superclass and at the same time strategy
#       classes that inherit such temperature routine classes Super SA -> Temperature SA -> Strategy SA

@dataclass
class MorpherResetSma(SimulatedAnnealingSpaceSearch):
    TEMPERATURE_STRATEGY_ID: str = field(default="MORPHER-RESET-SMA-RESET-SMA_", init=False)

    def __post_init__(self):
        super().__post_init__()

        # TODO maybe good to benchmark the same way we treat start t and items as cooling?
        self.START_TEMPERATURE = 100
        self.ITEMS_PER_TEMPERATRE = 100

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
            if self.temperature > self.FREEZING_TEMPERATRE:
                self.temperature = self.updateTemperature(self.temperature, acceptance_rate)

            # reheating
            if self.cost_sma_fast < self.cost_sma_slow and self.cost_sma_slow < self.cost_sma_fast + self.EPSILON and acceptance_rate < 0.1:
                self.temperature = self.START_TEMPERATURE
                self.cost_sma_slow = self.start_configuration_cost

            print(f"{int(running_since):4d}s A_RT: {acceptance_rate:2.2f}  C: {self.sol_cost:6d}  SMA {self.SMA_SLOW_ITEMS}: {self.cost_sma_slow:4.1f}  SMA {self.SMA_FAST_ITEMS}: {self.cost_sma_fast:4.1f}  T: {self.temperature:4.4f}", end='\r')

            self.iterations += 1
            self.cumulative_temp_routine_time += (time.process_time() - before_temperature_routine_time)
            
            self.cumulative_running_time += (time.process_time() - while_running_time)
