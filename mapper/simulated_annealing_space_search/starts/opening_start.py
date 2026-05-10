from dataclasses import field

from plots import save_plot_mappings_cgra
from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

class OpeningStart(SimulatedAnnealingSpaceSearch):
    START_ID: str = field(default="OPENING-START", init=False)

    # Shared functions
    def initial_sol_generator(self):
        """
        Random solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.
        """
        curr_node_pe: dict[int, int] = {}
        curr_pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        # invece di andare a caso guardare il neghbour con la funzione costo e
        # scegliere la posizione che minimizza il costo

        # Very naive: we place all pes on the same tile for each schedule
        middle = size // 2

        for t in self.schedule:
            for n in self.schedule[t]:
                curr_node_pe[n] = middle

        def is_node_overlapping(n: int, curr_node_pe: dict[int, int], neighbours: list[int]) -> bool:
            for nei in neighbours:
                if n != nei and curr_node_pe[n] == curr_node_pe[nei]:
                    return True
            return False

        # at random for each schedule we chose one of the adjacent PEs we continue until
        # all operations are on a single pe
        for t in sorted(self.schedule):
            for n in sorted(self.schedule[t]):
                while is_node_overlapping(n, curr_node_pe, self.schedule[t]):
                    cnpe = curr_node_pe[n]
                    adjacent_pes = [
                        pe for pe in [
                            cnpe + 1,
                            cnpe - 1,
                            cnpe + self.size_x,
                            cnpe - self.size_x
                        ] if 0 <= pe < size
                    ]

                    curr_node_pe[n] = self.randgen_start_configuration.choice(adjacent_pes)

        # build curr_pe_nodes
        for n, pe in curr_node_pe.items():
            if pe not in curr_pe_nodes:
                curr_pe_nodes[pe] = []
            curr_pe_nodes[pe].append(n)

        # apply to main variables
        self.node_pe = curr_node_pe
        self.pe_nodes = curr_pe_nodes

        print(f"Start solution cost: {self.cost_space_solution(self.node_pe)}")
        save_plot_mappings_cgra(self.node_pe, self.schedule, self.size_x, self.size_y, id="opening_start")
