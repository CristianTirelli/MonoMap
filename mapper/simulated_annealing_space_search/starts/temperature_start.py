from dataclasses import dataclass, field
import math

from plots import save_plot_mappings_cgra
from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

@dataclass
class TemperatureStart(SimulatedAnnealingSpaceSearch):
    START_ID: str = field(default="TEMPERATURE-START", init=False)

    overlapping_nodes: list[int] = field(default_factory=list, init=False)

    def nodes_pes_overlap_count(self, curr_node_pe: dict[int, int], build_overlapping_nodes: bool = False):
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
        # little messy, dataclass should ensure all init and post init are called but it is not the case
        self.overlapping_nodes = []

        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # unpositioned nodes
        unpositioned_nodes = list(range(self.dfg.number_of_nodes()))

        # mini-SA
        T = 10

        # Using MA has a problem: we start with a very very low cost of 0, then we ramp up,
        # and if we SA with all nodes within the CGRA is like we are doing the regular routine
        # then why do so here, I think starting at high temperature as too possibility to diverge
        # unless we ramp up the malues of cost to high levels

        # even overlap cost does not make sense: we have an empty CGRA nodes are not overlapping, nor will
        # also cost: no node is in the CGRA cost is zero at the beginning
        # such combinations make for immediate acceptange, basically random construction

        prev_coverlap: int = 0
        coverlap: int = self.nodes_pes_overlap_count(curr_node_pe)
        prev_csol: int = 0
        csol: int = 0

        OVERLAP_ID = 0
        UNPOSITIONED_ID = 1

        nunpositioned_nodes: int = len(unpositioned_nodes)

        plateau_times: int = 0
        while 0 != coverlap or nunpositioned_nodes != 0:

            items: int = 0
            while items < self.ITEMS_PER_TEMPERATRE:
                if coverlap == 0 and nunpositioned_nodes == 0:
                    break
                # select a random node

                choice_array_id = "NONE"
                choice_array = None
                if self.randgen_start_configuration.random() < 0.5:
                    choice_array_id = UNPOSITIONED_ID
                    choice_array = unpositioned_nodes
                else:
                    choice_array_id = OVERLAP_ID
                    choice_array = self.overlapping_nodes

                if len(choice_array) == 0:
                    choice_array = self.overlapping_nodes if choice_array_id == UNPOSITIONED_ID else unpositioned_nodes
                    choice_array_id = OVERLAP_ID if choice_array_id == UNPOSITIONED_ID else UNPOSITIONED_ID


                node_to_position = self.randgen_start_configuration.choice(choice_array)

                # we select a random position
                pe_to_position = self.randgen_start_configuration.randint(0, size - 1)

                # move there
                old_pe_position = curr_node_pe[node_to_position] if choice_array_id == OVERLAP_ID else None
                curr_node_pe[node_to_position] = pe_to_position

                # position by cost + overlap cost
                curr_csol = self.cost_space_solution(curr_node_pe, silent=True)
                curr_coverlap = self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=True)

                # we are always accepting right away
                # we change it to: see x items and take the best
                if curr_csol <= csol and curr_coverlap <= coverlap:
                    # accept right away
                    csol = curr_csol
                    coverlap = curr_coverlap
                    if choice_array_id == UNPOSITIONED_ID:
                        unpositioned_nodes.remove(node_to_position)
                else:
                    # accept by T
                    # we want to add much more weight to overlaps than solution cost
                    P = math.exp(- ((curr_csol - csol) ** 0.5 + (curr_coverlap - coverlap) ** 2) / T)

                    rnd = self.randgen_start_configuration.random()
                    if rnd < P:
                        csol = curr_csol
                        coverlap = curr_coverlap
                        if choice_array_id == UNPOSITIONED_ID:
                            unpositioned_nodes.remove(node_to_position)
                    else:
                        if choice_array_id == UNPOSITIONED_ID:
                            curr_node_pe.pop(node_to_position)
                        if choice_array_id == OVERLAP_ID:
                            curr_node_pe[node_to_position] = old_pe_position
                        coverlap = self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=True)

                nunpositioned_nodes = len(unpositioned_nodes)
                items += 1

            if csol == prev_csol and coverlap == prev_coverlap:
                plateau_times += 1

                if plateau_times == 50:
                    # simple spike reheating
                    T = 10
                    plateau_times = 0
            else:
                plateau_times = 0
            
            print(f"T: {T:3.4f} coverlap: {coverlap} prev: {prev_coverlap} csol: {csol} prev: {prev_csol} nunpositioned: {nunpositioned_nodes} PT: {plateau_times:2.0f}", end="\r")

            T *= 0.9
            prev_coverlap = coverlap
            prev_csol = csol


        # build curr_pe_nodes
        for n, pe in curr_node_pe.items():
            if pe not in curr_pe_nodes:
                curr_pe_nodes[pe] = []
            curr_pe_nodes[pe].append(n)

        # apply to main variables
        self.node_pe = curr_node_pe
        self.pe_nodes = curr_pe_nodes

        print(f"Start solution cost: {self.cost_space_solution(self.node_pe)}")
        save_plot_mappings_cgra(self.node_pe, self.schedule, self.size_x, self.size_y, id="temperature_start")


@dataclass
class TemperatureStartInverted(SimulatedAnnealingSpaceSearch):
    START_ID: str = field(default="TEMPERATURE-START-INVERTED", init=False)


    overlapping_nodes: list[int] = field(default_factory=list, init=False)

    def nodes_pes_overlap_count(self, curr_node_pe: dict[int, int], build_overlapping_nodes: bool = False):
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
        # little messy, dataclass should ensure all init and post init are called but it is not the case
        self.overlapping_nodes = []

        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # unpositioned nodes
        unpositioned_nodes = list(range(self.dfg.number_of_nodes()))

        # mini "bottom up" SA: from freezing T to hot T
        T: float = 0.00001

        OVERLAP_ID = 0
        UNPOSITIONED_ID = 1

        coverlap = 0
        csol = 0
        while 0 != self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=True) or len(unpositioned_nodes) != 0:
            # select array
            choice_array_id = "NONE"
            choice_array: list[int] = None
            if self.randgen_start_configuration.random() < 0.5:
                choice_array_id = UNPOSITIONED_ID
                choice_array = unpositioned_nodes
            else:
                choice_array_id = OVERLAP_ID
                choice_array = self.overlapping_nodes

            if len(choice_array) == 0:
                choice_array = self.overlapping_nodes if choice_array_id == UNPOSITIONED_ID else unpositioned_nodes
                choice_array_id = OVERLAP_ID if choice_array_id == UNPOSITIONED_ID else UNPOSITIONED_ID

            # select a random node
            node_to_position = self.randgen_start_configuration.choice(choice_array)

            # we select a random position
            pe_to_position = self.randgen_start_configuration.randint(0, size - 1)

            # move there
            old_pe_position = curr_node_pe[node_to_position] if choice_array_id == OVERLAP_ID else None
            curr_node_pe[node_to_position] = pe_to_position

            # position by cost + overlap cost
            curr_csol = self.cost_space_solution(curr_node_pe, silent=True)
            curr_coverlap = self.nodes_pes_overlap_count(curr_node_pe, build_overlapping_nodes=False)

            if curr_csol <= csol and curr_coverlap <= coverlap:
                # accept right away
                csol = curr_csol
                coverlap = curr_coverlap

                if choice_array_id == UNPOSITIONED_ID:
                    unpositioned_nodes.remove(node_to_position)
            else:
                # accept by T
                # we want to add much more weight to overlaps than solution cost
                P = math.exp(- ((curr_csol - csol) + (curr_coverlap - coverlap) ** 2) / T)

                rnd = self.randgen_start_configuration.random()
                if rnd < P:
                    csol = curr_csol
                    coverlap = curr_coverlap

                    if choice_array_id == UNPOSITIONED_ID:
                        unpositioned_nodes.remove(node_to_position)
                else:
                    if choice_array_id == UNPOSITIONED_ID:
                        curr_node_pe.pop(node_to_position)
                    if choice_array_id == OVERLAP_ID:
                        curr_node_pe[node_to_position] = old_pe_position

            print(f"T: {T:3.4f} coverlap: {coverlap} csol: {csol}", end="\r")

            T *= 1.0001



        # build curr_pe_nodes
        for n, pe in curr_node_pe.items():
            if pe not in curr_pe_nodes:
                curr_pe_nodes[pe] = []
            curr_pe_nodes[pe].append(n)

        # apply to main variables
        self.node_pe = curr_node_pe
        self.pe_nodes = curr_pe_nodes

        print(f"Start solution cost: {self.cost_space_solution(self.node_pe)}")
        save_plot_mappings_cgra(self.node_pe, self.schedule, self.size_x, self.size_y, id="temperature-inverted_start")
