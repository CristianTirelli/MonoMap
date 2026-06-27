# strategies
from simulated_annealing_space_search.strategies.poisson_routine_types import PoissonRoutineEnum
from simulated_annealing_space_search.strategies.worst_positioned_nodes_poisson import WorstPositionedNodesPoisson

from simulated_annealing_space_search.strategies.random_node_with_swap import RandomNodeWithSwap
from simulated_annealing_space_search.strategies.random_node_with_swap import RandomNodeWithSwapNewRoutine

from simulated_annealing_space_search.strategies.random_node import RandomNode
from simulated_annealing_space_search.strategies.visits_heuristic import RandomNodeWithSwapAndVisitsHeuristic

# routines
from simulated_annealing_space_search.routines.cooling import Cooling
from simulated_annealing_space_search.routines.cooling_reset_sma import CoolingResetSma
from simulated_annealing_space_search.routines.morpher_reset_sma import MorpherResetSma

from simulated_annealing_space_search.routines.cooling_reset_to_probability_sma import CoolingResetToProbabilityDynamicBestCostSma
from simulated_annealing_space_search.routines.cooling_reset_to_probability_sma import CoolingResetToProbabilityDynamicLearnedSma
from simulated_annealing_space_search.routines.cooling_reset_to_probability_sma import FixedCoolingResetToProbabilitySma

from simulated_annealing_space_search.routines.cooling_reset_to_probability_restart import CoolingResetToProbabilityDynamicBestCostSmaRestart

# start solution
from simulated_annealing_space_search.starts.random_start import RandomStart
from simulated_annealing_space_search.starts.opening_start import OpeningStart
from simulated_annealing_space_search.starts.greedy_start import GreedyStart
from simulated_annealing_space_search.starts.temperature_start import TemperatureStart
from simulated_annealing_space_search.starts.temperature_start import TemperatureStartInverted
from simulated_annealing_space_search.starts.temperature_start_centered import TemperatureStartCentered
from simulated_annealing_space_search.starts.temperature_start_centered import TemperatureStartCenteredInverted

# classic
class RandomNodeCooling(RandomNode, Cooling, RandomStart):
    pass

class RandomNodeCoolingResetSma(RandomNode, CoolingResetSma, RandomStart):
    pass

class RandomNodeMorpherResetSma(RandomNode, MorpherResetSma, RandomStart):
    pass

class RandomNodeWithSwapMorpherResetSma(RandomNodeWithSwap, MorpherResetSma, RandomStart):
    pass


# list selection, we do upgrade with spike
class WorstPositionedNodesPoissonCoolingResetSmaRandomStart(WorstPositionedNodesPoisson, CoolingResetSma, RandomStart):
    pass
# no spike
class WorstPositionedNodesPoissonCoolingRandomStart(WorstPositionedNodesPoisson, Cooling, RandomStart):
    pass

# new swap routine
class RandomNodeWithSwapNewRoutineCoolingResetSma(RandomNodeWithSwapNewRoutine, CoolingResetSma, RandomStart):
    pass


# with reset to fixed probability
class RandomNodeWithSwapFixedCoolingResetToProbabilitySma(RandomNodeWithSwap, FixedCoolingResetToProbabilitySma, RandomStart):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicLearnedSma(RandomNodeWithSwap, CoolingResetToProbabilityDynamicLearnedSma, RandomStart):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSma(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, RandomStart):

    def __post_init__(self):
        super().__post_init__()

        # Used for different model number tests
        # self.STRATEGY_ID = f"RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-{self.MODEL_NUMBER}-SAME-RUN"
        # self.STRATEGY_ID = f"RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-{self.MODEL_NUMBER}-SAME-RUN"
        # self.STRATEGY_ID = f"RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-{self.MODEL_NUMBER}-SAME-RUN"


# different start algorithms
class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaOpeningStart(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, OpeningStart):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaGreedyStart(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, GreedyStart):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaTemperatureStart(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, TemperatureStart):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaTemperatureStartInverted(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, TemperatureStartInverted):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaTemperatureStartCentered(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, TemperatureStartCentered):
    pass

class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaTemperatureStartCenteredInverted(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSma, TemperatureStartCenteredInverted):
    pass



# repeated start
class RandomNodeWithSwapCoolingResetToProbabilityDynamicBestCostSmaRestartRandomStart(RandomNodeWithSwap, CoolingResetToProbabilityDynamicBestCostSmaRestart, RandomStart):
    pass


# poisson pick, greedy start and reheating to probability
class WorstPositionedNodesPoissonCoolingResetToProbabilityDynamicBestCostSmaGreedyStart(WorstPositionedNodesPoisson, CoolingResetToProbabilityDynamicBestCostSma, GreedyStart):
    pass
# list selection, we do upgrade with spike
class WorstPositionedNodesPoissonCoolingResetToProbabilityDynamicBestCostSmaRandomStart(WorstPositionedNodesPoisson, CoolingResetToProbabilityDynamicBestCostSma, RandomStart):
    pass

# visits heuristic
class RandomNodeWithSwapAndVisitsHeuristicCoolingResetSma(RandomNodeWithSwapAndVisitsHeuristic, CoolingResetSma, RandomStart):
    pass
