import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

from os.path import join
from pathlib import Path

from sgd_over_dlns.dataset import generate_dataset
from sgd_over_dlns.network import LinearDiagonalNetwork
from sgd_over_dlns import plot

EXPERIMENT_OUTPUT_PATH = join(plot.OUTPUT_FOLDER, 'gain_shape')

def simulate_gains(n, d, beta, stochastic):
    batch_size = 1 if stochastic else None 
    gains = []
    for run in tqdm(range(100)):
        x, y = generate_dataset(n=n, d=d, beta=beta, rng=rng)
        net = LinearDiagonalNetwork(alpha=.1, lr=1e-3, dim=d, batch_size=batch_size, random_state=rng)
        net.train(x, y, iterations=10000)
        gains.append(net.gain)
    return gains

def plot_coordinates_and_average(filename, gains, color):
    mean_gain = np.mean(gains, axis=0)
    coordinates = range(1, len(mean_gain)+1)
    plt.clf()

    for gain in gains:
        plt.plot(coordinates, gain, alpha=0.1)  

    plt.plot(coordinates, mean_gain, linewidth=2, alpha=1.0, color=color) 
    plt.axvline(x=s+1, linestyle='--', c='black')

    plt.xlabel('Coordinate')
    plt.ylabel('Value')
    plt.xscale('log')

    plt.savefig(join(EXPERIMENT_OUTPUT_PATH, filename), format='png', dpi=1200)

if __name__ == '__main__':
    Path(EXPERIMENT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Experiment parameters
    n, d, s = 50, 100, 4
    rng = np.random.default_rng(seed=42)

    # Initialize beta
    beta = np.zeros(d)
    beta[0] = 100
    beta[1] = -100
    beta[2] = 100
    beta[3] = -100

    # Run experiments
    print('Running GD...')
    gains_gd = simulate_gains(n, d, beta, stochastic=False)
    print('Running SGD...')
    gains_sgd = simulate_gains(n, d, beta, stochastic=True)

    print('Saving images...')
    plot_coordinates_and_average('gain_shape_gd.png', gains_gd, 'blue')
    plot_coordinates_and_average('gain_shape_sgd.png', gains_sgd, 'red')