# TODO perfect implementation and test on single cases
# calculate how many solution there are given the current position of nodes, and moving one node at a time
# start random solution
# compute neighbour solution
# if solution is not accepted add the movement to the tabu list
# conintue this way until solution is found or tablu list blocks all moves
# if it does, increase temperature: at freezing tabu list has size as many possibilities there are, with more warm it has less, start empty

from dataclasses import dataclass, field
import math
import random
import time

from plots import saveFigTemperature
from simulated_annealing.TODO_simulated_annealing_tabu_space_search_morpher import SimulatedAnnealingTabuSpaceSearchMorpher

# NEIGHBOUR:    We take an operations at random and position it at random on all other available positions (its previous
#               position excluded)

@dataclass
class RandomNodeTabu(SimulatedAnnealingTabuSpaceSearchMorpher):
    STRATEGY_ID: str = field(default=SimulatedAnnealingTabuSpaceSearchMorpher.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE", init=False)

    # map of operation -> tabu positions
    tabu_list: dict[int, list[int]] = field(default_factory=dict, init=False)
    tabu_list_size: int = field(default=None, init=False)
    maximum_random_nodes_movement_possibilities: int = field(default=0, init=False)

    last_move: tuple[int, int] = field(default=None, init=False)

    # map of node -> number of scheduled nodes in the schedule the mapping nodes is from
    node_scheduled_nodes_count: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.maximum_random_nodes_movement_possibilities = self.compute_maximum_random_nodes_movement_possibilities()


    def compute_maximum_random_nodes_movement_possibilities(self) -> int:
        total_possibilities: int = 0
        size: int = self.size_x * self.size_y

        for t in self.schedule:
            operations_per_clock = len(self.schedule[t])

            for n in self.schedule[t]:
                self.node_scheduled_nodes_count[n] = operations_per_clock

            available_position_per_operation = size - operations_per_clock
            total_possibilities += available_position_per_operation * operations_per_clock
        return total_possibilities


    def is_operation_tabu(self, node: int, move_to_pe: int) -> bool:
        return move_to_pe in self.tabu_list[node]


    def add_operation_tabu(self, node: int, move_to_pe) -> bool:
        if node not in self.tabu_list:
            self.tabu_list[node] = []
        self.tabu_list[node].append(move_to_pe)
        self.tabu_list_size += 1


    def remove_operation_tabu(self):
        # removes one operation at random from tabu list
        key = random.choice(list(self.tabu_list.keys()))
        value_index = random.choice(len(self.tabu_list[key]))

        # remove it
        self.tabu_list[key][value_index] = self.tabu_list[key][-1]
        self.tabu_list[key].pop()
        self.tabu_list_size -= 1


    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """
        if self.tabu_list_size == self.maximum_random_nodes_movement_possibilities:
            raise AssertionError("Impossible to make a move")
        
        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        node_to_move: int = random.randint(0, len(self.node_pe))

        # do-while
        while True:
            node_to_move = random.randint(0, len(self.node_pe))

            if len(self.tabu_list[node_to_move]) < size - self.node_scheduled_nodes_count[node_to_move]:
                # node_to_move can pick another move
                break
        new_pe: int
        
        for t in self.schedule:
            if node_to_move in self.schedule[t]:
                swapping_with_node: int = -1

                # move at random
                # place it on remaining spots
                block_list = [self.node_pe[n] for n in self.schedule[t] if n != node_to_move]
                block_list.append(self.node_pe[node_to_move])

                # add tabu moves
                block_list.append(pe for pe in self.tabu_list[node_to_move])

                allow_list = [i for i in range(size) if i not in block_list]


                if len(allow_list) == 0:
                    # this given the current world view is a double move: two nodes move
                    # to a new position
                    # TODO: I dont consider the case, if the strategy is revealed effective
                    #       I can improve it
                    # TODO: it will probably break tabu list
                
                    # means all places are full
                    # we draw and swap with one that has the same
                    block_list.remove(self.node_pe[node_to_move])
                    new_pe = random.choice(block_list)
                    curr_pe = self.node_pe[node_to_move]

                    # will wlays be found
                    for n in self.schedule[t]:
                        if new_pe == self.node_pe[n]:
                            swapping_with_node = n

                    curr_node_pe[swapping_with_node] = curr_pe
                    curr_node_pe[node_to_move] = new_pe

                    if new_pe not in curr_pe_nodes:
                        curr_pe_nodes[new_pe] = []
                    curr_pe_nodes[new_pe].append(node_to_move)

                    if curr_pe not in curr_pe_nodes:
                        curr_pe_nodes[curr_pe] = []
                    curr_pe_nodes[curr_pe].append(swapping_with_node)
                else:
                    new_pe = random.choice(allow_list)

                    curr_node_pe[node_to_move] = new_pe

                    if new_pe not in curr_pe_nodes:
                        curr_pe_nodes[new_pe] = []
                    curr_pe_nodes[new_pe].append(node_to_move)

                for n in self.schedule[t]:
                    if n != node_to_move and n != swapping_with_node:
                        # maintain rest in schedule
                        pe = self.node_pe[n]
                        curr_node_pe[n] = pe

                        if pe not in curr_pe_nodes:
                            curr_pe_nodes[pe] = []
                        curr_pe_nodes[pe].append(n)
            else:
                # maintain rest
                for n in self.schedule[t]:
                    pe = self.node_pe[n]
                    curr_node_pe[n] = pe

                    if pe not in curr_pe_nodes:
                        curr_pe_nodes[pe] = []
                    curr_pe_nodes[pe].append(n)

        self.last_move = (node_to_move, new_pe)
        return (curr_node_pe, curr_pe_nodes)

    # Tabu Specific Simulated Annealing Search
    def simulatedAnnealingSearch(self) -> tuple[dict[int, int],  dict[int, list[int]], float]:
        print("*** START SA ROUTINE ***\n")
        start = time.time()

        self.temperature = self.START_TEMPERATURE

        # Generate starting random solution and evaluate it
        self.random_sol_generator()
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

        warmup_cost: int
        warming_up = False
        warming_up_step = 1

        iterations = 0
        while self.sol_cost != 0 and iterations < self.MAX_ITERATIONS:
            running_time = time.time() - start

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

                # naive: clear whole list
                self.tabu_list = {}
                self.tabu_list_size = 0

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

                    # naive: clear whole list
                    self.tabu_list = {}
                    self.tabu_list_size = 0

                    acceptance += 1
                else:
                    # solution is not accepted: we add it to tabu list based on temperature
                    # delta_E = c - self.sol_cost
                    # P = math.exp(- delta_E / self.temperature)
                    # is there a valid delta_E?

                    rnd = random.random()
                    P = self.temperature / self.START_TEMPERATURE
                    if rnd < P:
                        moved_node = self.last_move[0]
                        if moved_node not in self.tabu_list:
                            self.tabu_list[moved_node] = []
                        self.tabu_list[moved_node].append(self.last_move[1])

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
                    # we still have no intel here

                    # we should either empty the tabu list or free it a little as we walk up
                    self.tabu_list = {}
                    self.tabu_list_size = 0

                    # t increase
                    self.temperature += 0.001 / math.log(warming_up_step + math.e)

                    # based on temperature cap
                    if self.temperature >= self.TEMPERATURE_REFUEL_LIMIT:
                        warming_up = False

                    # based on better solutions found
                    if self.sol_cost < warmup_cost:
                        warming_up = False
                    warming_up_step += 1
                else:
                    # catch when tabu list is full
                    if self.tabu_list_size == self.maximum_random_nodes_movement_possibilities:
                        # we should either empty the tabu list or free it a little as we walk up
                        self.tabu_list = {}
                        self.tabu_list_size = 0

                        warming_up = True
                        warming_up_step = 1
                        warmup_cost = self.sol_cost

                    if self.temperature > self.FREEZING_TEMPERATURE:
                        self.temperature = self.updateTemperature(self.temperature, acceptance_rate)
                iterations += 1
            else:
                items += 1
        end = time.time()
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