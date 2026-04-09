
from dataclasses import dataclass, field


@dataclass
class RandomNodeWithSwap_OldNeighborRoutine():
    STRATEGY_ID: str = field(default="OLD-SWAP-ROUTINE", init=False)

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
