# (S)GD Over Diagonal Linear Networks

Authors: Alexandre Maranhão and Arthur Gallois

This repository is dedicated to reproducing the experiments from the paper [(S)GD Over Diagonal Linear Networks
](https://arxiv.org/abs/2302.08982). It was created as part of a Master's course and has no involvement or endorsement from the authors of the paper.

### Setup 

To setup the environment, install the requirements from `requirements.txt` using the environment manager of your choice. One possibility is with venv, by executing

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Experiments

To run the experiments, execute from the repository root

```bash
python -m sgd_over_dlns.experiments.1_varying_stepsize
python -m sgd_over_dlns.experiments.2_gain_shape
python -m sgd_over_dlns.experiments.3_edge_of_stability
```

The output images will be written in a folder called `output` under the repository root.

### References

- *M. Even, S. Pesme, S. Gunasekar, and N. Flammarion* (2023). **(S)GD over Diagonal Linear Networks: Implicit Regularisation, Large Stepsizes and Edge of Stability**. In: arXiv [cs.LG] [![DOI:10.48550/arXiv.2302.08982](https://zenodo.org/badge/DOI/10.48550/arXiv.2302.08982.svg)](https://doi.org/10.48550/arXiv.2302.08982)
