
import numpy as np

def format_possible_float_number(n):
    if not isinstance(n, float):
        return str(n)
    return str(round_to_first_digit(n)) if n < 0 else str(round(n, 2))

def round_to_first_digit(x):
    if x == 0: return 0
    precision = -int(np.floor(np.log10(abs(x))))
    return round(x, precision)
