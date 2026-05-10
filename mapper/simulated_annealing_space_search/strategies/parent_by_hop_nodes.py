
from dataclasses import dataclass, field

from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch


# TODO complete and assert

@dataclass
class RandomNodeWithSwapNewRoutine(SimulatedAnnealingSpaceSearch):
    HOPS = field(default=1, init=False)

    STRATEGY_ID = "PARENT-BY-HOP-NODE"

    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
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
        node_to_move: int = self.randgen_algorithm_run.randint(0, self.NNODES - 1)

        old_pe = curr_node_pe[node_to_move]

        # see if parent are there
        predecessors = self.directed_dfg.predecessors(node_to_move)
        predecessors_pe = []
        
        # collect on hop nodes from predecessors
        for p in predecessors:
            # predecessor pe
            p_pe = curr_node_pe[p]
            possible_pes = [
                p_pe + 1,
                p_pe - 1,
                p_pe + self.size_x,
                p_pe - self.size_x
            ]

            for ppe in possible_pes:
                if ppe not in predecessors_pe:
                    predecessors_pe.append(ppe)

        new_pe: int = self.randgen_algorithm_run.choice(predecessors_pe)
        
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
