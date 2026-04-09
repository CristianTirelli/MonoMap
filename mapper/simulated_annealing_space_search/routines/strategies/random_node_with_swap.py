from dataclasses import dataclass, field

# may lead to circular dependency, not the best solution
from plots import visit_heuristics_plot_run_graphs
from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

from simulated_annealing_space_search.routines.cooling_reset_sma import CoolingResetSma
from simulated_annealing_space_search.routines.morpher_reset_sma import MorpherResetSma
from simulated_annealing_space_search.routines.cooling_reset_to_probability_sma import CoolingResetToProbabilityDynamicSma, CoolingResetToProbabilityDynamicLearnedSma, FixedCoolingResetToProbabilitySma


class RandomNodeWithSwapMixin:
    def neighbour_sol_generator(self: SimulatedAnnealingSpaceSearch) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        node_to_move = self.randgen_algorithm_run.randint(0, len(self.node_pe) - 1)

        for t in self.schedule:
            if node_to_move in self.schedule[t]:
                # choose any PE besides its current pe
                pe_node_to_move = self.node_pe[node_to_move]
                allow_list = [pe for pe in range(size) if pe != pe_node_to_move]
                new_pe: int = self.randgen_algorithm_run.choice(allow_list)

                # check if collieds with any neighbour
                for n in self.schedule[t]:
                    if n != node_to_move:
                        if new_pe == self.node_pe[n]:
                            # give pe of node to move to node on new_pe
                            curr_node_pe[n] = pe_node_to_move
                            if pe_node_to_move not in curr_pe_nodes:
                                curr_pe_nodes[pe_node_to_move] = []
                            curr_pe_nodes[pe_node_to_move].append(n)
                        else:
                            # maintain n
                            pe = self.node_pe[n]
                            curr_node_pe[n] = pe

                            if pe not in curr_pe_nodes:
                                curr_pe_nodes[pe] = []
                            curr_pe_nodes[pe].append(n)

                # place at node_to_move at surely free new_pe
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


class RandomNodeWithSwapNewRoutineMixin:
    def neighbour_sol_generator(self: SimulatedAnnealingSpaceSearch) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        # to remember, new solution is much faster
        # curr_node_pe: dict[int, int] = copy.deepcopy(self.node_pe)
        # curr_pe_nodes: dict[int, list[int]] = copy.deepcopy(self.pe_nodes)
        # int are immutable copy is enough
        curr_node_pe: dict[int, int] = self.node_pe.copy()
        # need to copy one layer deeper
        curr_pe_nodes: dict[int, list[int]] = {k: v[:] for k, v in self.pe_nodes.items()}

        # select node to move
        node_to_move: int = self.randgen_algorithm_run.randint(0, self.nnodes - 1)
        # draw node
        # we could also redraw until not its pe
        last_idx_pe: int = self.size_x * self.size_y - 1
        new_pe: int = self.randgen_algorithm_run.randint(0, last_idx_pe)

        old_pe = curr_node_pe[node_to_move]
        while new_pe == old_pe:
            new_pe = self.randgen_algorithm_run.randint(0, last_idx_pe)
        
        # same pe 
        if old_pe == new_pe:
            # same pe
            return (curr_node_pe, curr_pe_nodes)
        # we are sure we'll remove
        curr_pe_nodes[old_pe].remove(node_to_move)

        t = self.node_schedule_t[node_to_move]
        neighbors = self.schedule[t]
        
        for n in neighbors:
            if curr_node_pe[n] == new_pe:
                # swap to do
                curr_pe_nodes[new_pe].remove(n)
                curr_node_pe[n] = old_pe
                curr_pe_nodes[old_pe].append(n)
                break
            
        # assignment
        curr_node_pe[node_to_move] = new_pe

        if new_pe not in curr_pe_nodes:
            curr_pe_nodes[new_pe] = []
        curr_pe_nodes[new_pe].append(node_to_move)
        return  (curr_node_pe, curr_pe_nodes)



@dataclass
class RandomNodeWithSwapFixedCoolingResetToProbabilitySma(RandomNodeWithSwapMixin, FixedCoolingResetToProbabilitySma):
    STRATEGY_ID: str = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.STRATEGY_ID = self.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WITH-SWAP"

        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.STRATEGY_ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.STRATEGY_ID)


@dataclass
class RandomNodeWithSwapCoolingResetToProbabilityDynamicLearnedSma(RandomNodeWithSwapMixin, CoolingResetToProbabilityDynamicLearnedSma):
    STRATEGY_ID: str = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.STRATEGY_ID = self.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WITH-SWAP"

        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.STRATEGY_ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.STRATEGY_ID)


@dataclass
class RandomNodeWithSwapCoolingResetToProbabilityDynamicSma(RandomNodeWithSwapMixin, CoolingResetToProbabilityDynamicSma):
    STRATEGY_ID: str = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.STRATEGY_ID = self.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WITH-SWAP"

        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.STRATEGY_ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.STRATEGY_ID)



@dataclass
class RandomNodeWithSwapNewRoutineCoolingResetSma(RandomNodeWithSwapNewRoutineMixin, CoolingResetSma):
    STRATEGY_ID: str = field(default=CoolingResetSma.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WITH-SWAP-NEW-ROUTINE-FASTER-COPY", init=False)
    
    def __post_init__(self):
        super().__post_init__()

        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.STRATEGY_ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.STRATEGY_ID)


@dataclass
class RandomNodeWithSwapMorpherResetSma(RandomNodeWithSwapMixin, MorpherResetSma):
    STRATEGY_ID: str = field(default=MorpherResetSma.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WITH-SWAP", init=False)
    
    def __post_init__(self):
        super().__post_init__()

        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.STRATEGY_ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.STRATEGY_ID)



# TODO complete algo
@dataclass
class RandomNodeWithSwapAndVisitsHeuristicCoolingResetSma(CoolingResetSma):
    STRATEGY_ID: str = field(default=CoolingResetSma.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WITH-SWAP-VISITS", init=False)

    # map of time -> node -> # visits
    # map of time -> node -> malus
    time_node_pe_visits: dict[str, dict[int, list[int]]] = field(default_factory=dict, init=False)
    time_node_pe_malus: dict[str, dict[int, list[int]]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        super().__post_init__()

        # warmup visits dictionary
        for t in self.schedule:
            self.time_node_pe_visits[t] = {}
            self.time_node_pe_malus[t] = {}

            for n in self.schedule[t]:
                # based on start solution
                self.time_node_pe_visits[t][n] = [1 for _ in range(self.size_x * self.size_y)]
                self.time_node_pe_malus[t][n] = [0 for _ in range(self.size_x * self.size_y)]
                # self.time_node_pe_visits[t][n][self.node_pe[n]] = 1
        
        if self.BENCHMARK:
            self.BENCHMARK.set_algorithm_type(self.STRATEGY_ID)
        if self.RECORDER:
            self.RECORDER.set_algorithm_type(self.STRATEGY_ID)


    def __temperature_reset__(self):
        """
        We update malus table
        """
        super().__temperature_reset__()

        # iterate over all dependencies
        for e in self.dfg.edges:
            source = e[0]
            destination = e[1]

            # should always be present
            if source not in self.node_pe:
                raise AssertionError("[__temperature_reset__]: source is not present in node_pe")
            if destination not in self.node_pe:
                raise AssertionError("[__temperature_reset__]: destination is not present in node_pe")
            ps = self.node_pe[source]
            pd = self.node_pe[destination]

            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x):
                cost = (self.pe_distance(pd, ps) ** 2) * 10

                # we add a malus equal to the edge cost
                self.time_node_pe_malus[self.node_schedule_t[source]][source][ps] += cost
                self.time_node_pe_malus[self.node_schedule_t[destination]][destination][pd] += cost
            else:
                # should consider to give back to good positioned nodes
                # consider there might be some good some bad for same node
                self.time_node_pe_malus[self.node_schedule_t[source]][source][ps] *= 0.75
                self.time_node_pe_malus[self.node_schedule_t[destination]][destination][pd] *= 0.75

    
    def __extra_plots__(self, extra_plot_path: str | None):
        super().__extra_plots__(extra_plot_path)

        visit_heuristics_plot_run_graphs(self.size_x, self.size_y, self.time_node_pe_visits, self.time_node_pe_malus, extra_plot_path)


    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # node at random
        node_to_move = self.randgen_algorithm_run.randint(0, len(self.node_pe) - 1)

        for t in self.schedule:
            if node_to_move in self.schedule[t]:
                # choose any PE besides its current pe
                pe_node_to_move = self.node_pe[node_to_move]

                # out of the available pe lists we give each a probability
                # based on visits, the few the more probability
                pe_visits = self.time_node_pe_visits[t][node_to_move]
                pe_malus = self.time_node_pe_malus[t][node_to_move]

                # we compute proabilities
                total_value = 0
                for i in range(size):
                    total_value += pe_visits[i] + pe_malus[i]

                if total_value == 0:
                    weights = [1.0 / size for _ in range(size)]
                else:
                    # inverse as to make more vistis and malues give less chance
                    weights = [1 - (pe_visits[pe] + pe_malus[pe]) / total_value for pe in range(size)]


                # select with relative visit probabilities
                new_pe = self.randgen_algorithm_run.choices(
                    population=range(size), 
                    weights=weights, 
                    k=1
                )[0]

                # add visit count
                # self.time_node_pe_visits[t][node_to_move][new_pe] += 1

                # check if collides with any neighbour
                for n in self.schedule[t]:
                    if n != node_to_move:
                        if new_pe == self.node_pe[n]:
                            # give pe of node to move to node on new_pe
                            curr_node_pe[n] = pe_node_to_move
                            if pe_node_to_move not in curr_pe_nodes:
                                curr_pe_nodes[pe_node_to_move] = []
                            curr_pe_nodes[pe_node_to_move].append(n)
                        else:
                            # maintain n
                            pe = self.node_pe[n]
                            curr_node_pe[n] = pe

                            if pe not in curr_pe_nodes:
                                curr_pe_nodes[pe] = []
                            curr_pe_nodes[pe].append(n)

                # place at node_to_move at surely free new_pe
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
