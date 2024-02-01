import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

from os.path import join
from pathlib import Path

from sgd_over_dlns.dataset import generate_dataset
from sgd_over_dlns.network import LinearDiagonalNetwork
from sgd_over_dlns import plot

EXPERIMENT_OUTPUT_PATH = join(plot.OUTPUT_FOLDER, 'varying_stepsize')

def simulate(stepsizes, stochastic):
    batch_size = 1 if stochastic else None 
    test_losses = np.zeros(len(stepsizes))
    gains = [ ]
    has_converged = np.zeros(len(stepsizes), dtype=bool)

    for k, lr in tqdm(enumerate(stepsizes), total=len(stepsizes)):
        net = LinearDiagonalNetwork(alpha=.1, lr=lr, dim=d, batch_size=batch_size, random_state=rng)
        net.train(x, y, iterations=100000)
        test_loss = net.get_loss(x_test, y_test)
        test_losses[k] = test_loss
        gains.append(net.gain)

    has_converged = ~np.isnan(test_losses)

    last_convergent = np.argwhere(has_converged == True)[-1, 0]
    lr_max = stepsizes[last_convergent]

    gain_scales = np.array([np.linalg.norm(gain, ord=1) for gain in gains])

    return test_losses, has_converged, lr_max, gain_scales

if __name__ == '__main__':
    Path(EXPERIMENT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Experiment parameters
    d, n, s = 30, 20, 3
    rng = np.random.default_rng(seed=1234)

    # Initialize beta
    beta = np.zeros(d)
    nonzero_ix = list(range(s))
    beta[nonzero_ix] = rng.normal(0, 1, size=s)

    # Generate datasets
    x, y = generate_dataset(n, d, beta, rng=rng)
    x_test, y_test = generate_dataset(n, d, beta, rng=rng)

    # For each learning rate
    stepsizes = np.power(10, np.linspace(-2, 1, 30))

    print('Running GD...')
    test_losses_gd, has_converged_gd, lr_max_gd, gains_scale_gd = simulate(stepsizes, stochastic=False)
    print('Running SGD...')
    test_losses_sgd, has_converged_sgd, lr_max_sgd, gains_scale_sgd = simulate(stepsizes, stochastic=True)

    # Plot test losses
    plt.clf()
    plt.plot(stepsizes[has_converged_gd], test_losses_gd[has_converged_gd], c='b', label='GD')
    plt.plot(stepsizes[has_converged_sgd], test_losses_sgd[has_converged_sgd], c='r', label='SGD')
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(x=lr_max_gd, linestyle='--', c='b')
    plt.axvline(x=lr_max_sgd, linestyle='--', c='r')

    plt.xlabel('Stepsize')
    plt.ylabel('Test loss')
    plt.legend()

    plt.savefig(join(EXPERIMENT_OUTPUT_PATH, 'image_testloss_stepsizes.png'), format='png', dpi=1200)

    # Plot Gain scale
    plt.clf()
    plt.plot(stepsizes[has_converged_gd], gains_scale_gd[has_converged_gd], c='b', label='GD')
    plt.plot(stepsizes[has_converged_sgd], gains_scale_sgd[has_converged_sgd], c='r', label='SGD')
    plt.xscale('log')
    plt.axvline(x=lr_max_gd, linestyle='--', c='b')
    plt.axvline(x=lr_max_sgd, linestyle='--', c='r')

    plt.xlabel('Stepsize')
    plt.ylabel('Gain Norm')
    plt.legend()

    plt.savefig(join(EXPERIMENT_OUTPUT_PATH, 'gain_scale.png'), format='png', dpi=1200)
