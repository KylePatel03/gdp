import numpy as np

def init_weighted_max_sat_weights(N,seed):
    rng = np.random.default_rng(seed=seed)
    weights = rng.choice([-1,1],size=(N,N))
    weights = weights - np.diag(np.diag(weights))
    weights = np.tril(weights) + np.tril(weights,-1).T
    return weights

def init_weighted_max_sat_states(N,seed):
    rng = np.random.default_rng(seed=seed)
    states = rng.choice([-1,1],size=N)
    return states