from dataclasses import dataclass, field
import random

from networkx import DiGraph

from simulated_annealing.strategies.poisson_routine_types import PoissonRoutineEnum
from simulated_annealing.simulated_annealing_space_search_morpher_warmup_sma import SimulatedAnnealingSpaceSearchMorpherWarmupSma

import numpy as np

# COST:         The cost is computed by looping over all dependency edges and check if they are respected,
#               if they are not we add the distance squared to the cost to discourage further away dependencies.
#               While calculating the cost we keep track of distances of nodes whose edges are not connected
#               and we build a list that holds operations indexes based on their cost and their dependencies
#               where incoming dependencies weight more than outgoing dependencies, ordered from more costly
#               to less costly.
#               A variation might be that first we order by distance cost and then we order by dependencies cost.
#               Indicated as "Then" class.

# NEIGHBOUR:    The routine that computes the neighbour solution is as follows: We take the schedule and the
#               previous solution and a list of nodes based on distance cost, which has been computed inside
#               COST. Given these information we draw a random node following three different kind of procedures:
#                   1. A poisson distribution with a fixed lambda
#                   2. A poisson distributio with a proportional lambda to the array size
#                   3. A poisson distribution that calculates lambda proportional to the current temperature
#               The node darw following the distribution on the worst sorted order of the operation array
#               is the one that will be moved. We maintain the current solution for all schedules that do not
#               have the selected node and for the one that has we select a random PE position from the one
#               available minus the current PE of such node.


@dataclass
class WorstPositionedMostDependenciesNodesPoisson(SimulatedAnnealingSpaceSearchMorpherWarmupSma):
    STRATEGY_ID: str = field(default=None, init=False)
    BASE_STRATEGY: str = field(default="WORST-POSITIONED-MOST-DEPENDENCIES-NODE_", init=False)

    directed_dfg: DiGraph

    poisson_routine_type: PoissonRoutineEnum

    fixed_lambda_value: int | None = field(kw_only=True, default=None),
    proportional_lambda_percentage: float | None = field(kw_only=True, default=None),

    worst_nodes: list[int] = field(default_factory=list, init=False),

    def __post_init__(self):
        """
        Used to check constructor variables validity, constructor is handled by dataclass

        Poisson rotuine can be set to three different kind:

            -   Fixed lambda, where lambda is passed in as integers and used as fixed value for the whole run duration
            -   Proportional lambda percentage, where the proportion manages where lambda is positioned proportionally to the nodes array length
                For example, a proportion of 0.0 will set lambda to be 0, a proportion of 0.5 will set lambda to be at half the size of nodes, and so on.
            -   Temperature lambda, where the lambda follows the current temperature, the higher it is the more right positioned the distribution is, which
                will move towards providing neighbour that move correct nodes, so worst solution
        """
        if not self.directed_dfg:
            raise AssertionError(f"Directed DFG graph is needed")

        match self.poisson_routine_type:
            case PoissonRoutineEnum.FIXED_LAMBDA:
                if self.fixed_lambda_value < 0:
                    raise AssertionError("Lambda cannot be negative")
                
                self.STRATEGY_ID = SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + f"{self.BASE_STRATEGY}POISSON-FIXED-{self.fixed_lambda_value}"
            case PoissonRoutineEnum.PROPORTIONAL_LAMBDA:
                if self.proportional_lambda_percentage < 0 or 1 < self.proportional_lambda_percentage:
                    raise AssertionError(f"Invalid proportional poisson lambda percentage: {self.proportional_lambda_percentage}")
            
                self.STRATEGY_ID = SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + f"{self.BASE_STRATEGY}POISSON-PROPORTIONAL-{self.proportional_lambda_percentage}"
            case PoissonRoutineEnum.TEMPERATURE_LAMBDA:
                self.STRATEGY_ID = SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + f"{self.BASE_STRATEGY}POISSON-PROPORTIONAL-TEMPERATURE"
            case _:
                raise AssertionError(f"Poisson routine not supported: {self.poisson_routine_type}")

    def cost_space_solution(self, curr_node_pe: dict[int, int]) -> int:
        """
        In addition to calculating the cost based on Manhattan distance we compute
        the list of worst positioned nodes based on maximum distance
        """
        cost = 0

        worst_nodes: dict[int, int] = {}

        for e in self.dfg.edges:
            source = e[0]
            destination = e[1]

            if source not in curr_node_pe:
                raise AssertionError("source is not present in node_pe")
            if destination not in curr_node_pe:
                raise AssertionError("destination is not present in node_pe")
            ps = curr_node_pe[source]
            pd = curr_node_pe[destination]

            d = 0
            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x):
                d = self.pe_distance(self.arch, pd, ps)
                # print(f"Source operation {source} at pe {pd} to operation {destination} at pe {ps} has distance {d}")
                cost += d ** 2

                worst_nodes[source] = d + self.directed_dfg.in_degree(source) * 20 + self.directed_dfg.out_degree(source) * 10
                worst_nodes[destination] = d + self.directed_dfg.in_degree(destination) * 20 + self.directed_dfg.out_degree(destination) * 10
            else:
                if source not in worst_nodes:
                    worst_nodes[source] = self.directed_dfg.in_degree(source) * 20 + self.directed_dfg.out_degree(source) * 10
                if destination not in worst_nodes:
                    worst_nodes[destination] = self.directed_dfg.in_degree(destination) * 20 + self.directed_dfg.out_degree(destination) * 10

        self.worst_nodes = [k for k, _ in sorted(worst_nodes.items(), key=lambda item: item[1], reverse=True)]
        return cost

    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Constructs neighbour by selecting the worst node from the worst positioned nodes following
        a poisson distribution and moving it at a random posiiton. Its previous position is excluded.
        """

        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # from poisson distribution get node
        # depending on routine type
        nnodes = len(self.worst_nodes)
        node_to_move: int
        match self.poisson_routine_type:
            case PoissonRoutineEnum.FIXED_LAMBDA:
                # fixed lambda: we clamp in array range
                node_to_move_idx = np.clip(np.random.poisson(self.fixed_lambda_value), 0, nnodes - 1)
                node_to_move = self.worst_nodes[node_to_move_idx]
            case PoissonRoutineEnum.PROPORTIONAL_LAMBDA:
                lmda: int = nnodes * self.proportional_lambda_percentage
                node_to_move_idx = np.clip(np.random.poisson(lmda), 0, nnodes - 1)
                node_to_move = self.worst_nodes[node_to_move_idx]
            case PoissonRoutineEnum.TEMPERATURE_LAMBDA:
                # the function for which we map temperature: [100, freezing (~0.001)] = [x1, x2]
                # to lambda: [0, nnodes - 1] = [a, b]
                # using: y = a + (b - a) * f(t, t1, t2), where f returns the temperature in [0, 1] range following a curve, then we normalize it
                # where f is non linear, we use gamma, like gamma correction of the graphics class for reality-display intensity: f(t) = \frac{x - x2}{x1 - x2}^\gamma, where \gamma = 0.45
                # we dont want to reach completely the beginning of the array as to leave some hill climb possibilities
                lmda: float
                if self.temperature < self.FREEZING_TEMPERATURE:
                    lmda = 0
                else:
                    lmda = (nnodes - 1) * (((self.temperature - self.FREEZING_TEMPERATURE) / (self.START_TEMPERATURE - self.FREEZING_TEMPERATURE)) ** 0.45)
                node_to_move_idx = np.clip(np.random.poisson(lmda), 0, nnodes - 1)
                node_to_move = self.worst_nodes[node_to_move_idx]

        # Q: what exactly is lambda and what does it manage?
        # lambda manages the curve center: https://www.geeksforgeeks.org/python/numpy-random-poisson-in-python/
        # in our case the x-axis holds the worst nodes indexes and the lower the lambda is
        # the more probable it will be to pick up a bad positioned node.

        for t in self.schedule:
            if node_to_move in self.schedule[t]:
                # move at random
                for n in self.schedule[t]:
                    if n != node_to_move:
                        # maintain rest in schedule
                        pe = self.node_pe[n]
                        curr_node_pe[n] = pe

                        if pe not in curr_pe_nodes:
                            curr_pe_nodes[pe] = []
                        curr_pe_nodes[pe].append(n)

                # place it on remaining spots
                block_list = [curr_node_pe[n] for n in self.schedule[t] if n != node_to_move]
                block_list.append(self.node_pe[node_to_move])
                allow_list = [i for i in range(size) if i not in block_list]

                new_pe = random.choice(allow_list)

                curr_node_pe[node_to_move] = new_pe

                if new_pe not in curr_pe_nodes:
                    curr_pe_nodes[new_pe] = []
                curr_pe_nodes[new_pe].append(node_to_move)
            else:
                # maintain rest
                for n in self.schedule[t]:
                    pe = self.node_pe[n]
                    curr_node_pe[n] = pe

                    if pe not in curr_pe_nodes:
                        curr_pe_nodes[pe] = []
                    curr_pe_nodes[pe].append(n)

        return (curr_node_pe, curr_pe_nodes)


@dataclass
class WorstPositionedThenMostDependenciesNodesPoisson(SimulatedAnnealingSpaceSearchMorpherWarmupSma):
    STRATEGY_ID: str = field(default=None, init=False)
    BASE_STRATEGY: str = field(default="WORST-POSITIONED-THEN-MOST-DEPENDENCIES-NODE_", init=False)

    directed_dfg: DiGraph

    poisson_routine_type: PoissonRoutineEnum

    fixed_lambda_value: int | None = field(kw_only=True, default=None),
    proportional_lambda_percentage: float | None = field(kw_only=True, default=None),

    worst_nodes: list[int] = field(default_factory=list, init=False),

    def __post_init__(self):
        """
        Used to check constructor variables validity, constructor is handled by dataclass

        Poisson rotuine can be set to three different kind:

            -   Fixed lambda, where lambda is passed in as integers and used as fixed value for the whole run duration
            -   Proportional lambda percentage, where the proportion manages where lambda is positioned proportionally to the nodes array length
                For example, a proportion of 0.0 will set lambda to be 0, a proportion of 0.5 will set lambda to be at half the size of nodes, and so on.
            -   Temperature lambda, where the lambda follows the current temperature, the higher it is the more right positioned the distribution is, which
                will move towards providing neighbour that move correct nodes, so worst solution
        """
        if not self.directed_dfg:
            raise AssertionError(f"Directed DFG graph is needed")

        match self.poisson_routine_type:
            case PoissonRoutineEnum.FIXED_LAMBDA:
                if self.fixed_lambda_value < 0:
                    raise AssertionError("Lambda cannot be negative")
                
                self.STRATEGY_ID = SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + f"{self.BASE_STRATEGY}POISSON-FIXED-{self.fixed_lambda_value}"
            case PoissonRoutineEnum.PROPORTIONAL_LAMBDA:
                if self.proportional_lambda_percentage < 0 or 1 < self.proportional_lambda_percentage:
                    raise AssertionError(f"Invalid proportional poisson lambda percentage: {self.proportional_lambda_percentage}")
            
                self.STRATEGY_ID = SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + f"{self.BASE_STRATEGY}POISSON-PROPORTIONAL-{self.proportional_lambda_percentage}"
            case PoissonRoutineEnum.TEMPERATURE_LAMBDA:
                self.STRATEGY_ID = SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + f"{self.BASE_STRATEGY}POISSON-PROPORTIONAL-TEMPERATURE"
            case _:
                raise AssertionError(f"Poisson routine not supported: {self.poisson_routine_type}")

    def cost_space_solution(self, curr_node_pe: dict[int, int]) -> int:
        """
        In addition to calculating the cost based on Manhattan distance we compute
        the list of worst positioned nodes based on maximum distance
        """
        cost = 0

        worst_nodes: dict[int, tuple[int, int]] = {}

        for e in self.dfg.edges:
            source = e[0]
            destination = e[1]

            if source not in curr_node_pe:
                raise AssertionError("source is not present in node_pe")
            if destination not in curr_node_pe:
                raise AssertionError("destination is not present in node_pe")
            ps = curr_node_pe[source]
            pd = curr_node_pe[destination]

            d = 0
            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x):
                d = self.pe_distance(self.arch, pd, ps)
                # print(f"Source operation {source} at pe {pd} to operation {destination} at pe {ps} has distance {d}")
                cost += d ** 2

                worst_nodes[source] = (d, self.directed_dfg.in_degree(source) * 20 + self.directed_dfg.out_degree(source) * 10)
                worst_nodes[destination] = (d, self.directed_dfg.in_degree(destination) * 20 + self.directed_dfg.out_degree(destination) * 10)
            else:
                if source not in worst_nodes:
                    worst_nodes[source] = (0, self.directed_dfg.in_degree(source) * 20 + self.directed_dfg.out_degree(source) * 10)
                if destination not in worst_nodes:
                    worst_nodes[destination] = (0, self.directed_dfg.in_degree(destination) * 20 + self.directed_dfg.out_degree(destination) * 10)

        self.worst_nodes = [k for k, _ in sorted(worst_nodes.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)]
        return cost

    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Constructs neighbour by selecting the worst node from the worst positioned nodes following
        a poisson distribution and moving it at a random posiiton. Its previous position is excluded.
        """

        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # from poisson distribution get node
        # depending on routine type
        nnodes = len(self.worst_nodes)
        node_to_move: int
        match self.poisson_routine_type:
            case PoissonRoutineEnum.FIXED_LAMBDA:
                # fixed lambda: we clamp in array range
                node_to_move_idx = np.clip(np.random.poisson(self.fixed_lambda_value), 0, nnodes - 1)
                node_to_move = self.worst_nodes[node_to_move_idx]
            case PoissonRoutineEnum.PROPORTIONAL_LAMBDA:
                lmda: int = nnodes * self.proportional_lambda_percentage
                node_to_move_idx = np.clip(np.random.poisson(lmda), 0, nnodes - 1)
                node_to_move = self.worst_nodes[node_to_move_idx]
            case PoissonRoutineEnum.TEMPERATURE_LAMBDA:
                # the function for which we map temperature: [100, freezing (~0.001)] = [x1, x2]
                # to lambda: [0, nnodes - 1] = [a, b]
                # using: y = a + (b - a) * f(t, t1, t2), where f returns the temperature in [0, 1] range following a curve, then we normalize it
                # where f is non linear, we use gamma, like gamma correction of the graphics class for reality-display intensity: f(t) = \frac{x - x2}{x1 - x2}^\gamma, where \gamma = 0.45
                # we dont want to reach completely the beginning of the array as to leave some hill climb possibilities
                lmda: float = nnodes * (((self.temperature - self.FREEZING_TEMPERATURE) / (self.START_TEMPERATURE - self.FREEZING_TEMPERATURE)) ** 0.45)
                node_to_move_idx = np.clip(np.random.poisson(lmda), 0, nnodes - 1)
                node_to_move = self.worst_nodes[node_to_move_idx]

        # Q: what exactly is lambda and what does it manage?
        # lambda manages the curve center: https://www.geeksforgeeks.org/python/numpy-random-poisson-in-python/
        # in our case the x-axis holds the worst nodes indexes and the lower the lambda is
        # the more probable it will be to pick up a bad positioned node.

        for t in self.schedule:
            if node_to_move in self.schedule[t]:
                # move at random
                for n in self.schedule[t]:
                    if n != node_to_move:
                        # maintain rest in schedule
                        pe = self.node_pe[n]
                        curr_node_pe[n] = pe

                        if pe not in curr_pe_nodes:
                            curr_pe_nodes[pe] = []
                        curr_pe_nodes[pe].append(n)

                # place it on remaining spots
                block_list = [curr_node_pe[n] for n in self.schedule[t] if n != node_to_move]
                block_list.append(self.node_pe[node_to_move])
                allow_list = [i for i in range(size) if i not in block_list]

                new_pe = random.choice(allow_list)

                curr_node_pe[node_to_move] = new_pe

                if new_pe not in curr_pe_nodes:
                    curr_pe_nodes[new_pe] = []
                curr_pe_nodes[new_pe].append(node_to_move)
            else:
                # maintain rest
                for n in self.schedule[t]:
                    pe = self.node_pe[n]
                    curr_node_pe[n] = pe

                    if pe not in curr_pe_nodes:
                        curr_pe_nodes[pe] = []
                    curr_pe_nodes[pe].append(n)

        return (curr_node_pe, curr_pe_nodes)
