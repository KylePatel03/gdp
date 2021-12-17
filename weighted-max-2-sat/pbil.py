import numpy as np
from collections import Counter
from math import ceil

class PBIL:
    def __init__(self, weights):
        self.weights = weights
        self.N = self.weights.shape[0]
        # Each entry stores the probability of a variable taking value 1
        self.P = np.full((1,self.N), 0.5)
        self.P0 = self.P.copy()
    
    def cost(self,sample):
        energy = -0.5 * (sample @ self.weights) @ sample.T
        return energy.flatten()[0]
    
    def sample(self):
        s = np.empty((1,self.N),dtype=int)
        for i,p1 in enumerate(self.P[0]):
            s[0][i] = np.random.choice([-1,1],size=None,p=[1-p1,p1])
        return s
    
    def update_P(self,samples,learning_rate):
        empirical_p = self._empirical_probabilities(samples)
        self.P = learning_rate * self.P + (1-learning_rate) * empirical_p
    
    def _empirical_probabilities(self,samples):
        p = np.empty((1,self.N))
        num_samples = len(samples)
        for i in range(self.N):
            count = Counter(sample[0][i] for sample in samples)
            p[0][i] = count[1] / num_samples
        return p

    # iterations: Number of times the distribution is sampled and updated
    # population_size: Total number of samples drawn during each iteration
    # sample_frac: The fraction (0,1] of the population used to update distribution
    # learning_rate: [0,1]
    def run(self,iterations,population_size,sample_frac,learning_rate):
        sample_size = ceil(sample_frac * population_size)
        energies = []
        for _ in range(iterations):
            samples = [self.sample() for _ in range(population_size)]
            sample_energy_pairs = [(x,self.cost(x)) for x in samples]
            sample_energy_pairs.sort(key=lambda x: x[1], reverse=False)

            best_samples = []
            best_samples_energy = []
            for sample, energy in sample_energy_pairs[:sample_size]:
                best_samples.append(sample)
                best_samples_energy.append(energy)
            self.update_P(best_samples,learning_rate)
            energies.append(best_samples_energy)
        return energies
    
    def run_mean_energy(self,p):
        # Unpack parameters
        iterations = p[0]
        population_size = p[1]
        sample_frac = p[2]
        learning_rate = p[3]

        sample_size = ceil(sample_frac * population_size)
        best_samples_energies = []
        for _ in range(iterations):
            samples = [self.sample() for _ in range(population_size)]
            sample_energy_pairs = [(x,self.cost(x)) for x in samples]
            sample_energy_pairs.sort(key=lambda x: x[1], reverse=False)

            best_samples = [p[0] for p in sample_energy_pairs[:sample_size]]
            best_samples_energies = [p[1] for p in sample_energy_pairs[:sample_size]]
            self.update_P(best_samples,learning_rate)
        return np.mean(best_samples_energies)