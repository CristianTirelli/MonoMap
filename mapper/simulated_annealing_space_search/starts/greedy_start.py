from dataclasses import dataclass, field
import math

from plots import save_plot_mappings_cgra
from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

@dataclass
class GreedyStart(SimulatedAnnealingSpaceSearch):
    START_ID: str = field(default="GREEDY-START", init=False)

    overlapping_nodes: list[int] = field(default_factory=list, init=False)

    def nodes_pes_overlap_count(self, curr_node_pe: dict[int, int], build_overlapping_nodes: bool = False):
        # improve: if you iterate over each schedule time t, collect pe from dictionary,for each duplicate +1, and exra +1
        # after when all introduction is written
        # and all implementation without this exact piece
        overlaps = 0
        if build_overlapping_nodes:
            self.overlapping_nodes.clear()

        for n in curr_node_pe:
            t = self.NODE_SCHEDULE_T[n]

            for nneighbor in self.schedule[t]:
                if nneighbor != n:
                    if nneighbor in curr_node_pe and curr_node_pe[nneighbor] == curr_node_pe[n]:
                        overlaps += 1

                        # update list: costly, for now ok
                        if build_overlapping_nodes and nneighbor not in self.overlapping_nodes:
                            self.overlapping_nodes.append(nneighbor)
                        if build_overlapping_nodes and n not in self.overlapping_nodes:
                            self.overlapping_nodes.append(n)

        return int(overlaps / 2)

    # Shared functions
    def initial_sol_generator(self):
        """
        Random solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.
        """
        # pre dataclass constructor init
        self.overlapping_nodes = []

        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # I position all nodes in the middle pe
        middle = size // 2

        for n in range(self.dfg.number_of_nodes()):
            curr_node_pe[n] = middle



        # i iterate until the overlapping cost is zero
        ocount = self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=True)
        while 0 < ocount:
            # I select one random overlapping node
            node_to_move = self.randgen_start_configuration.choice(self.overlapping_nodes)

            # I iterate over all pes and select the one that minimizes the cost function
            # while stricly decreasing overlap count
            # Note: at some point we have to eat some cost to get nodes overlap to zero
            # we modify solution each iteration as it must find a step
            best_pe = 0

            best_pe_sol_cost = math.inf
            best_ocount = ocount
            for pe in range(0, size):
                curr_node_pe[node_to_move] = pe

                pe_sol_cost = self.cost_space_solution(curr_node_pe)
                pe_ocount = self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=False)
                if pe_sol_cost <= best_pe_sol_cost and pe_ocount < best_ocount:
                    best_pe = pe

            curr_node_pe[node_to_move] = best_pe
            ocount = self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=True)



        # build curr_pe_nodes
        for n, pe in curr_node_pe.items():
            if pe not in curr_pe_nodes:
                curr_pe_nodes[pe] = []
            curr_pe_nodes[pe].append(n)

        # apply to main variables
        self.node_pe = curr_node_pe
        self.pe_nodes = curr_pe_nodes

        print(f"Start solution cost: {self.cost_space_solution(self.node_pe)}")
        save_plot_mappings_cgra(self.node_pe, self.schedule, self.size_x, self.size_y, id="greedy_start")
