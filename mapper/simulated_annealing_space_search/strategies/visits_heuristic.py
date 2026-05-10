from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

from plots import visit_heuristics_plot_run_graphs

# TODO refine
@dataclass
class RandomNodeWithSwapAndVisitsHeuristic(SimulatedAnnealingSpaceSearch):
    STRATEGY_ID: str = field(default="RANDOM-NODE-WITH-SWAP-VISITS", init=False)

    # map of time -> node -> # visits
    # map of time -> node -> malus
    time_node_pe_visits: dict[str, dict[int, list[int]]] = field(default_factory=dict, init=False)
    time_node_pe_malus: dict[str, dict[int, list[int]]] = field(default_factory=dict, init=False)


    NO_MOVEMENT: int = field(default=-1, init=False)

    moved_node: int = field(default=None, init=False)
    moved_node_old_pe_position: int = field(default=None, init=False)

    swapped_with_node: int = field(default=None, init=False)


    def __post_init__(self):
        super().__post_init__()

        size = self.size_x * self.size_y

        # warmup visits dictionary
        for t in self.schedule:
            self.time_node_pe_visits[t] = {}
            self.time_node_pe_malus[t] = {}

            for n in self.schedule[t]:
                self.time_node_pe_visits[t][n] = [0 for _ in range(size)]
                self.time_node_pe_malus[t][n] = [0 for _ in range(size)]


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
                self.time_node_pe_malus[self.NODE_SCHEDULE_T[source]][source][ps] += cost
                self.time_node_pe_malus[self.NODE_SCHEDULE_T[destination]][destination][pd] += cost
            else:
                # should consider to give back to good positioned nodes
                # consider there might be some good some bad for same node
                self.time_node_pe_malus[self.NODE_SCHEDULE_T[source]][source][ps] *= 0.75
                self.time_node_pe_malus[self.NODE_SCHEDULE_T[destination]][destination][pd] *= 0.75

    
    def __extra_plots__(self, extra_plot_path: str | None):
        super().__extra_plots__(extra_plot_path)

        visit_heuristics_plot_run_graphs(self.size_x, self.size_y, self.time_node_pe_visits, self.time_node_pe_malus, extra_plot_path)


    def neighbour_sol_generator(self):
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        curr_node_pe: dict[int, int] = {}

        size = self.size_x * self.size_y

        # node at random
        node_to_move = self.randgen_algorithm_run.randint(0, self.NNODES - 1)

        # choose any PE besides its current pe
        pe_node_to_move = self.node_pe[node_to_move]

        t = self.NODE_SCHEDULE_T[node_to_move]

        # out of the available pe lists we give each a probability
        # based on visits, the few the more probability
        pe_visits = self.time_node_pe_visits[t][node_to_move]
        pe_malus = self.time_node_pe_malus[t][node_to_move]

        # we compute proabilities
        total_value = 0
        for i in range(size):
            if i != pe_node_to_move:
                total_value += pe_visits[i] + pe_malus[i]

        if total_value == 0:
            weights = [1.0 / size for pe in range(size) if pe != pe_node_to_move]
        else:
            # inverse as to make more vistis and malues give less chance
            weights = [1 - (pe_visits[pe] + pe_malus[pe]) / total_value for pe in range(size) if pe != pe_node_to_move]


        # select with relative visit probabilities
        new_pe = self.randgen_algorithm_run.choices(
            population=range(size), 
            weights=weights, 
            k=1
        )[0]

        # add visit count
        # self.time_node_pe_visits[t][node_to_move][new_pe] += 1

        # check if collides with any neighbour
        for neigh in self.schedule[t]:
            if neigh != node_to_move and new_pe == self.node_pe[neigh]:
                    # give pe of node to move to node on new_pe
                    curr_node_pe[neigh] = pe_node_to_move

                    # undo_neighbour_sol_generator
                    self.swapped_with_node = neigh
                    break

        # place at node_to_move at surely free new_pe
        curr_node_pe[node_to_move] = new_pe
        # undo_neighbour_sol_generator
        self.moved_node = node_to_move
        self.moved_node_old_pe_position = pe_node_to_move


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



    def old_neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
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
