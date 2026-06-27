from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch
from simulated_annealing_space_search.strategies.poisson_routine_types import PoissonRoutineEnum

import numpy as np

# COST:         The cost is computed by looping over all dependency edges and check if they are respected,
#               if they are not we add the distance squared to the cost to discourage further away dependencies.
#               While calculating the cost we keep track of distances of nodes whose edges are not connected
#               and we build a list that holds operations indexes based on their cost, from worst to best.

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
class WorstPositionedNodesPoisson(SimulatedAnnealingSpaceSearch):
    STRATEGY_ID: str = field(default=None, init=False)
    BASE_STRATEGY: str = field(default="WORST-POSITIONED-NODE", init=False)


    # poisson
    poisson_routine_type: PoissonRoutineEnum

    poisson_randgen_algorithm_run: np.random.Generator | None = field(default=None, init=False),

    # # passed in based on poisson_routine_type
    fixed_lambda_value: int | None = field(kw_only=True, default=None),
    proportional_lambda_percentage: float | None = field(kw_only=True, default=None),

    worst_nodes: list[int] = field(default_factory=list, init=False)


    # undo solution
    NO_MOVEMENT: int = field(default=-1, init=False)

    moved_node: int = field(default=None, init=False)
    moved_node_old_pe_position: int = field(default=None, init=False)

    swapped_with_node: int = field(default=None, init=False)


    def __post_init__(self):
        """
        Used to check passed in constructor values, construction is handled by dataclass

        Poisson rotuine can be set to three different kind:

            -   Fixed lambda, where lambda is passed in as integers and used as fixed value for the whole run duration
            -   Proportional lambda percentage, where the proportion manages where lambda is positioned proportionally to the nodes array length
                For example, a proportion of 0.0 will set lambda to be 0, a proportion of 0.5 will set lambda to be at half the size of nodes, and so on.
            -   Temperature lambda, where the lambda follows the current temperature, the higher it is the more right positioned the distribution is, which
                will move towards providing neighbour that move correct nodes, so worst solution
        """
        super().__post_init__()

        # poisson rng with same seed as main class
        self.poisson_randgen_algorithm_run = np.random.default_rng(self.seed_algorithm_run)

        match self.poisson_routine_type:
            case PoissonRoutineEnum.FIXED_LAMBDA:
                if self.fixed_lambda_value < 0:
                    raise AssertionError("Lambda cannot be negative")
                
                self.STRATEGY_ID = f"{self.BASE_STRATEGY}_POISSON-FIXED-{self.fixed_lambda_value}"
            case PoissonRoutineEnum.PROPORTIONAL_LAMBDA:
                if self.proportional_lambda_percentage < 0 or 1 < self.proportional_lambda_percentage:
                    raise AssertionError(f"Invalid proportional poisson lambda percentage: {self.proportional_lambda_percentage}")
            
                self.STRATEGY_ID = f"{self.BASE_STRATEGY}_POISSON-PROPORTIONAL-{self.proportional_lambda_percentage}"
            case PoissonRoutineEnum.TEMPERATURE_LAMBDA:
                self.STRATEGY_ID = f"{self.BASE_STRATEGY}_POISSON-PROPORTIONAL-TEMPERATURE"
            case _:
                raise AssertionError(f"Poisson routine not supported: {self.poisson_routine_type}")

        self.moved_node = self.NO_MOVEMENT
        self.moved_node_old_pe_position = self.NO_MOVEMENT
        self.swapped_with_node = self.NO_MOVEMENT


    def cost_space_solution(self, curr_node_pe: dict[int, int], p: str | None = None, silent: bool = False) -> int:
        """
        In addition to calculating the cost based on Manhattan distance we compute
        the list of worst positioned nodes based on maximum distance
        """
        cost = 0

        if p:
            print(f"size_x: {self.size_x}, size_y: {self.size_y}")

        worst_nodes: dict[int, int] = {}

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

            d = 0
            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x, True if p is not None else False):
                d = self.pe_distance(pd, ps)
                cost += d ** 2

                if source not in worst_nodes or worst_nodes[source] < d:
                    worst_nodes[source] = d
                if destination not in worst_nodes or worst_nodes[destination] < d:
                    worst_nodes[destination] = d
            else:
                if source not in worst_nodes:
                    worst_nodes[source] = 0
                if destination not in worst_nodes:
                    worst_nodes[destination] = 0

        if p:
            print(f"{p} final cost: {cost}")
        self.worst_nodes = [k for k, _ in sorted(worst_nodes.items(), key=lambda item: item[1], reverse=True)]
        return cost


    def neighbour_sol_generator(self) -> dict[int, int]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        size = self.size_x * self.size_y


        self.randgen_algorithm_run

        # from poisson distribution get node
        # depending on routine type
        match self.poisson_routine_type:
            case PoissonRoutineEnum.FIXED_LAMBDA:
                # fixed lambda: we clamp in array range
                node_to_move_idx = np.clip(self.poisson_randgen_algorithm_run.poisson(self.fixed_lambda_value), 0, self.NNODES - 1)
                self.moved_node = self.worst_nodes[node_to_move_idx]
            case PoissonRoutineEnum.PROPORTIONAL_LAMBDA:
                lmda: int = (self.NNODES - 1) * self.proportional_lambda_percentage
                node_to_move_idx = np.clip(self.poisson_randgen_algorithm_run.poisson(lmda), 0, self.NNODES - 1)
                self.moved_node = self.worst_nodes[node_to_move_idx]
            case PoissonRoutineEnum.TEMPERATURE_LAMBDA:
                lmda: float
                lmda = (self.NNODES - 1) * ((self.temperature / self.START_TEMPERATURE) ** 0.45)

                node_to_move_idx = np.clip(self.poisson_randgen_algorithm_run.poisson(lmda), 0, self.NNODES - 1)
                self.moved_node = self.worst_nodes[node_to_move_idx]
            case _:
                raise AssertionError(f"Can't pick a node to move: Poisson routine not supported: {self.poisson_routine_type}")
        # only node selection chnages and uses poisson
    
        t = self.NODE_SCHEDULE_T[self.moved_node]

        block_list = [self.curr_node_pe[neigh] for neigh in self.schedule[t] if neigh != self.moved_node]
        block_list.append(self.node_pe[self.moved_node])
        allow_list = [i for i in range(size) if i not in block_list]

        self.moved_node_old_pe_position = self.curr_node_pe[self.moved_node]
        new_pe: int
        if len(allow_list) == 0:
            # means all places are full
            # we draw and swap with one that has already a pe
            # but we removee our own position
            block_list.remove(self.moved_node_old_pe_position)
            new_pe = self.randgen_algorithm_run.choice(block_list)

            # will wlays be found: the operation we are swapping with
            for neigh in self.schedule[t]:
                if new_pe == self.curr_node_pe[neigh]:
                    self.swapped_with_node = neigh

            self.curr_node_pe[self.swapped_with_node] = self.moved_node_old_pe_position
            self.curr_node_pe[self.moved_node] = new_pe
        else:
            new_pe = self.randgen_algorithm_run.choice(allow_list)
            self.curr_node_pe[self.moved_node] = new_pe

    def undo_neighbour_sol_generator(self):
        if self.moved_node != self.NO_MOVEMENT:
            if self.swapped_with_node != self.NO_MOVEMENT:
                self.curr_node_pe[self.swapped_with_node] = self.curr_node_pe[self.moved_node]
                self.curr_node_pe[self.moved_node] = self.moved_node_old_pe_position
            else:
                self.curr_node_pe[self.moved_node] = self.moved_node_old_pe_position

        self.moved_node = self.NO_MOVEMENT
        self.moved_node_old_pe_position = self.NO_MOVEMENT
        self.swapped_with_node = self.NO_MOVEMENT
