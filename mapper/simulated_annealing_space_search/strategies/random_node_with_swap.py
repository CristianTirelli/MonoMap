from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

@dataclass
class RandomNodeWithSwap(SimulatedAnnealingSpaceSearch):
    STRATEGY_ID: str = field(default="RANDOM-NODE-WITH-SWAP", init=False)


    NO_MOVEMENT: int = field(default=-1, init=False)

    moved_node: int = field(default=None, init=False)
    moved_node_old_pe_position: int = field(default=None, init=False)

    swapped_with_node: int = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()
        
        self.moved_node = self.NO_MOVEMENT
        self.moved_node_old_pe_position = self.NO_MOVEMENT
        self.swapped_with_node = self.NO_MOVEMENT

    def neighbour_sol_generator(self) -> dict[int, int]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        size = self.size_x * self.size_y

        node_to_move = self.randgen_algorithm_run.randint(0, self.NNODES - 1)

        pe_node_to_move = self.curr_node_pe[node_to_move]
        allow_list = [pe for pe in range(size) if pe != pe_node_to_move]
        new_pe: int = self.randgen_algorithm_run.choice(allow_list)

        t = self.NODE_SCHEDULE_T[node_to_move]

        # check if it collides with any neighbour
        for neigh in self.schedule[t]:
            if neigh != node_to_move and new_pe == self.curr_node_pe[neigh]:
                # give pe of node to move to node on new_pe
                self.curr_node_pe[neigh] = pe_node_to_move

                # undo_neighbour_sol_generator
                self.swapped_with_node = neigh
                break
        self.curr_node_pe[node_to_move] = new_pe
        
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


@dataclass
class RandomNodeWithSwapNewRoutine(SimulatedAnnealingSpaceSearch):
    STRATEGY_ID: str = field(default="RANDOM-NODE-WITH-SWAP-NEW-ROUTINE", init=False)


    NO_MOVEMENT: int = field(default=-1, init=False)

    moved_node: int = field(default=None, init=False)
    moved_node_old_pe_position: int = field(default=None, init=False)

    swapped_with_node: int = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()
        
        self.moved_node = self.NO_MOVEMENT
        self.moved_node_old_pe_position = self.NO_MOVEMENT
        self.swapped_with_node = self.NO_MOVEMENT

    def neighbour_sol_generator(self) -> dict[int, int]:
        """
        Neighbour solution generator.
        The function should use an existing solution to construct a neighbouring solution to traverse the solution space in the vicinity of the solution used.

        Generates a solution by moving one random node to a random position within its schedule

        :return: A random solution, composed by node_pe and pe_nodes dictionaries. The solution may be valid or invalid.
        :rtype: tuple[dict[int, int], dict[int, list[int]]]
        """ 
        # select node to move
        node_to_move: int = self.randgen_algorithm_run.randint(0, self.NNODES - 1)
        # draw node
        # we could also redraw until it is not its pe
        last_idx_pe: int = self.size_x * self.size_y - 1
        new_pe: int = self.randgen_algorithm_run.randint(0, last_idx_pe)

        old_pe = self.curr_node_pe[node_to_move]
        # same pe
        if old_pe == new_pe:
            return

        t = self.NODE_SCHEDULE_T[node_to_move]
        for neigh in self.schedule[t]:
            if self.curr_node_pe[neigh] == new_pe:
                # swap
                self.curr_node_pe[neigh] = old_pe
                
                # undo_neighbour_sol_generator
                self.swapped_with_node = neigh
                break
            
        # assignment
        self.curr_node_pe[node_to_move] = new_pe
        # undo_neighbour_sol_generator
        self.moved_node = node_to_move
        self.moved_node_old_pe_position = old_pe

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

        t = self.NODE_SCHEDULE_T[node_to_move]
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
