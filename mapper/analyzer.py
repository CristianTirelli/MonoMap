


import numpy as np
import matplotlib.pyplot as plt


def read_csv(file_path: str):
    data = np.genfromtxt(file_path, delimiter=',')
    data = data[~np.isnan(data)]


if __name__ == "__main__":
    # Routine to analize CSV file
    pass