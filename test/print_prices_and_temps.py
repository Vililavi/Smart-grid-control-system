import os
import numpy as np
import pandas as pd


def get_max_and_min_vals():
    curr_path = os.getcwd()
    data_folder = os.path.join(os.path.dirname(curr_path), "data")
    wind = pd.read_csv(os.path.join(data_folder, "wind_generation.csv"))
    up = pd.read_csv(os.path.join(data_folder, "up_regulation.csv"))
    down = pd.read_csv(os.path.join(data_folder, "down_regulation.csv"))
    prices_and_temps = np.load(os.path.join(data_folder, "default_price_and_temperatures.npy"))

    wind = wind['Wind power generation - hourly data']
    up = up['Up-regulating price in the Balancing energy market']
    down = down['Down-regulation price in the Balancing energy market']

    print(f"min wind gen: {wind.min()}, max wind gen: {wind.max()}")
    print(f"min up price: {up.min()}, max up price: {up.max()}")
    print(f"min down price: {down.min()}, max down price: {down.max()}")
    print(f"min price: {min(prices_and_temps[:, 0])}, max price: {max(prices_and_temps[:, 0])}")
    print(f"min temp: {min(prices_and_temps[:, 1])}, max temp: {max(prices_and_temps[:, 1])}")

    print(f"\nwind mean: {wind.mean()}, stdev: {wind.std()}")
    print(f"up mean: {up.mean()}, stdev: {up.std()}")
    print(f"down mean: {down.mean()}, stdev: {down.std()}")
    print(f"price mean: {np.mean(prices_and_temps[:, 0])}, stdev: {np.std(prices_and_temps[:, 0])}")
    print(f"temp mean: {np.mean(prices_and_temps[:, 1])}, stdev: {np.std(prices_and_temps[:, 1])}")
    base_hourly_loads = np.array([
        0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
        0.3, 0.5, 0.6, 0.6, 0.5, 0.5,
        0.5, 0.4, 0.4, 0.6, 0.8, 1.4,
        1.2, 0.9, 0.8, 0.6, 0.5, 0.4
    ])
    print(f"loads mean: {np.mean(base_hourly_loads)}, stdev: {np.std(base_hourly_loads)}")


def main():
    curr_path = os.getcwd()
    parent_folder = os.path.dirname(curr_path)
    data = np.load(os.path.join(parent_folder, "data", "default_price_and_temperatures.npy"))
    print(data.shape)
    print(data)


if __name__ == "__main__":
    # main()
    get_max_and_min_vals()
