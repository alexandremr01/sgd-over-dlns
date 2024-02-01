import numpy as np 
import matplotlib.pyplot as plt

from os.path import join
from pathlib import Path

from sgd_over_dlns.dataset import generate_dataset
from sgd_over_dlns.network import LinearDiagonalNetwork
from sgd_over_dlns import plot

EXPERIMENT_OUTPUT_PATH = join(plot.OUTPUT_FOLDER, 'edge_of_stability')

def plot_loss(eos_loss_history, flow_loss_history):
    filename = join(EXPERIMENT_OUTPUT_PATH, 'edge_of_stability_loss.png')
    plt.clf()
    plt.plot(eos_loss_history, c='orange', label='Edge of Stability')
    plt.plot(flow_loss_history, c='green', label='Gradient Flow')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    # plt.title('Loss of GD in Eos')
    plt.legend()
    plt.savefig(filename, format='png', dpi=1200)

def plot_coordinates(beta_history_eos, beta_history_flow):
    filename = join(EXPERIMENT_OUTPUT_PATH, 'edge_stability_coordinates.png')
    plt.clf()
    plt.plot(beta_history_eos[:, 0], c='orange', label='Supp Coordinates')
    plt.plot(beta_history_eos[:, 1], c='orange')

    plt.plot(beta_history_flow[:, 0], c='green', linestyle='--')
    plt.plot(beta_history_flow[:, 1], c='green', label='Gradient Flow', linestyle='--')
    plt.plot(beta_history_eos[:, 2], c='blue', alpha=0.2, label='Other coordinates')

    for i in range(3, d):
        plt.plot(beta_history_eos[:, i], c='blue', alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.xscale('log')
    plt.savefig(filename, format='png', dpi=1200)

if __name__ == '__main__':
    Path(EXPERIMENT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Experiment parameters
    n, d, s = 50, 100, 2
    rng = np.random.default_rng(seed=42)

    # Initialize beta
    beta = np.zeros(d)
    beta[0] = 100
    beta[1] = -100

    # Generate datasets
    x, y = generate_dataset(n=n, d=d, beta=beta, rng=rng)

    # Run experiment
    print('Running Gradient Flow...')
    net_flow = LinearDiagonalNetwork(
        alpha=.1, 
        lr=5e-3, 
        dim=d, 
        batch_size=None, 
        random_state=rng, 
        store_trajectory=True
    )
    net_flow.train(x, y, iterations=10000)
    beta_history_flow = np.array(net_flow.beta_history)

    print('Running Edge of Stability...')
    net_eos = LinearDiagonalNetwork(
        alpha=.1, 
        lr=1.75e-2, 
        dim=d, 
        batch_size=None, 
        random_state=rng, 
        store_trajectory=True
    )
    net_eos.train(x, y, iterations=10000)
    beta_history_eos = np.array(net_eos.beta_history)

    # Plot results
    print('Saving images...')
    plot_loss(net_eos.loss_history, net_flow.loss_history)
    plot_coordinates(beta_history_eos, beta_history_flow)
