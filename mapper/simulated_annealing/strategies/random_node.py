from dataclasses import dataclass, field
import random

from simulated_annealing.simulated_annealing_space_search_morpher import SimulatedAnnealingSpaceSearchMorpher
from simulated_annealing.simulated_annealing_space_search_morpher_warmup_sma import SimulatedAnnealingSpaceSearchMorpherWarmupSma
from simulated_annealing.simulated_annealing_space_search_temperature_sma import SimulatedAnnealingSpaceSearchTemperatureSma
from simulated_annealing.simulated_annealing_space_search_cooling import SimulatedAnnealingSpaceSearchCooling

# NEIGHBOUR:    We take an operations at random and position it at random on all other available positions (its previous
#               position excluded)

@dataclass
class RandomNode(SimulatedAnnealingSpaceSearchMorpher):
    STRATEGY_ID: str = field(default=SimulatedAnnealingSpaceSearchMorpher.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE", init=False)

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

        node_to_move = random.randint(0, len(self.node_pe))

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
                    new_pe = random.choice(block_list)
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
                    new_pe = random.choice(allow_list)

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


@dataclass
class RandomNodeWarmupSma(SimulatedAnnealingSpaceSearchMorpherWarmupSma):
    STRATEGY_ID: str = field(default=SimulatedAnnealingSpaceSearchMorpher.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WARMUP-SMA", init=False)

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

        node_to_move = random.randint(0, len(self.node_pe))

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
                    new_pe = random.choice(block_list)
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
                    new_pe = random.choice(allow_list)

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


@dataclass
class RandomNodeTemperatureSma(SimulatedAnnealingSpaceSearchTemperatureSma):
    STRATEGY_ID: str = field(default=SimulatedAnnealingSpaceSearchMorpher.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WARMUP-SMA", init=False)

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

        node_to_move = random.randint(0, len(self.node_pe))

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
                    new_pe = random.choice(block_list)
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
                    new_pe = random.choice(allow_list)

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


@dataclass
class RandomNodeCooling(SimulatedAnnealingSpaceSearchCooling):
    STRATEGY_ID: str = field(default=SimulatedAnnealingSpaceSearchMorpher.TEMPERATURE_STRATEGY_ID + "RANDOM-NODE-WARMUP-SMA", init=False)

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

        node_to_move = random.randint(0, len(self.node_pe))

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
                    new_pe = random.choice(block_list)
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
                    new_pe = random.choice(allow_list)

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
