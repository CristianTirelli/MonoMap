from dataclasses import dataclass, field
import random



# COST:         The cost is computed by looping over all dependency edges and check if they are respected,
#               if they are not we add the distance squared to the cost to discourage further away dependencies.
#               While calculating the cost we keep track of distances of nodes whose edges are not connected
#               and we build a list that holds operations indexes based on their cost, from worst to best.

# NEIGHBOUR:    The routine that computes the neighbour solution is as follows: We take the schedule and the
#               previous solution and a list of nodes based on distance cost, which has been computed inside
#               COST. Given these information we always draw the worst possible node. We maintain the current solution
#               for all schedules that do not have the selected node and for the one that has we select a random
#               PE position from the one available minus the current PE of such node.


@dataclass
class WorstPositionedNode():
    STRATEGY_ID: str = field(default=SimulatedAnnealingSpaceSearchMorpherWarmupSma.TEMPERATURE_STRATEGY_ID + "WORST-POSITIONED-NODE", init=False)

    worst_nodes: list[int] = field(default_factory=list, init=False)

    # could be abstraced along with worst_positioned_nodes_poisson
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

            ps = curr_node_pe[source]
            pd = curr_node_pe[destination]

            d = 0
            # if not connected means that pd, ps are int that they are not adjacent
            if not self.isConnected(pd, ps, self.size_y, self.size_x):
                d = self.pe_distance(self.arch, pd, ps)
                # print(f"Source operation {source} at pe {pd} to operation {destination} at pe {ps} has distance {d}")
                cost += d ** 2

                worst_nodes[source] = d
                worst_nodes[destination] = d
            else:
                if source not in worst_nodes:
                    worst_nodes[source] = 0
                if destination not in worst_nodes:
                    worst_nodes[destination] = 0

        self.worst_nodes = [k for k, _ in sorted(worst_nodes.items(), key=lambda item: item[1], reverse=True)]
        return cost

    def neighbour_sol_generator(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Constructs neighbour by selecting the worst node from the worst positioned nodes and moving it at a random
        posiiton. Its previous position is excluded.
        """

        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        node_to_move = self.worst_nodes[0]

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
