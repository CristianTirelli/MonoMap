from simulated_annealing_space_search.simulated_annealing_space_search import SimulatedAnnealingSpaceSearch

class RandomStart(SimulatedAnnealingSpaceSearch):
    START_ID: str = "RANDOM-START"

    # Generation of a random initial solution
    def initial_sol_generator(self):
        """
        Initial solution generator.

        It assumes that the schedule does not schedule more instructions than available PEs.
        For each schedule it positions instructions at random in available PEs.

        This implementation generates a random initial solution
        """
        node_pe: dict[int, int] = {}
        pe_nodes: dict[int, list[int]] = {}

        size = self.size_x * self.size_y

        for t in sorted(self.schedule):
            available_pes = [i for i in range(size)]

            for node in sorted(self.schedule[t]):
                pe = self.randgen_start_configuration.choice(available_pes)
                
                if pe not in pe_nodes:
                    pe_nodes[pe] = []
                pe_nodes[pe].append(node)

                if node not in node_pe:
                    node_pe[node] = pe
                else:
                    print(f"[RandomStart] should not happend, node: {node}, pe: {pe}")

                available_pes.remove(pe)

        # apply to main variables
        self.node_pe = node_pe
        self.pe_nodes = pe_nodes
