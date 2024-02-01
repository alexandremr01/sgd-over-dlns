import numpy as np

def generate_dataset(n, d, beta, rng):
    x = rng.multivariate_normal(np.zeros(d), cov=np.eye(d), size=n)
    y = x.dot(beta)
    return x, y