
from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

@dataclass
class RandomNode(SimulatedAnnealingSpaceSearch):
    STRATEGY_ID: str = field(default="RANDOM-NODE", init=False)

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

        self.moved_node = self.randgen_algorithm_run.randint(0, self.NNODES - 1)
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


    def old_neighbour_sol_generator(self) -> dict[int, int]:
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
                swapping_with_node: int = -1

                # move at random
                # place it on remaining spots
                block_list = [self.node_pe[n] for n in self.schedule[t] if n != node_to_move]
                block_list.append(self.node_pe[node_to_move])
                allow_list = [i for i in range(size) if i not in block_list]

                new_pe: int
                if len(allow_list) == 0:
                    # means all places are full
                    # we draw and swap with one that has the same
                    block_list.remove(self.node_pe[node_to_move])
                    new_pe = self.randgen_algorithm_run.choice(block_list)
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
                    new_pe = self.randgen_algorithm_run.choice(allow_list)

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
        return (curr_node_pe, curr_pe_nodes)
