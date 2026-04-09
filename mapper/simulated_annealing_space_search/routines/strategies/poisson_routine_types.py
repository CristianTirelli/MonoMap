from enum import Enum

class PoissonRoutineEnum(Enum):
    FIXED_LAMBDA: int = 0
    PROPORTIONAL_LAMBDA: int = 1
    TEMPERATURE_LAMBDA: int = 2
