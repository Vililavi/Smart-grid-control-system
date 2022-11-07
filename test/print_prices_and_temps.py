import os
import numpy as np


def main():
    curr_path = os.getcwd()
    parent_folder = os.path.dirname(curr_path)
    data = np.load(os.path.join(parent_folder, "data", "default_price_and_temperatures.npy"))
    print(data)


if __name__ == "__main__":
    main()
