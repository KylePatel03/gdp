import numpy as np
import time
from helper import init_weighted_max_sat_weights
from json import dump
from os.path import join

class PBIL:
    def __init__(self,N,seed_weights):
        self.N = N
        self.seed_weights = seed_weights
        self.weights = init_weighted_max_sat_weights(N,seed_weights)
        # Each entry stores the probability of a variable taking value 1
        self.P = self._init_P()
        self.P0 = self.P.copy()
    
    def _init_P(self):
        return np.full((1,self.N),0.5)
    
    def cost(self,sample):
        energy = -0.5 * (sample @ self.weights) @ sample.T
        return energy.flatten()[0]
    
    def sample(self):
        s = np.empty((1,self.N),dtype=int)
        for i,p1 in enumerate(self.P[0]):
            s[0][i] = np.random.choice([-1,1],size=None,p=[1-p1,p1])
        return s
    
    def run(self,iterations,population_size,learning_rate):
        iteration_energies = []
        iteration_times = []
        for i in range(iterations):
            start_time = time.time()
            # Generate population_size number of candidate solutions
            samples = [self.sample() for _ in range(population_size)]
            sample_energies = [self.cost(x) for x in samples]
            # Store the sample with the best fitness
            best_sample_index = int(np.argmin(sample_energies))
            best_sample = samples[best_sample_index]
            # Convert -1 to 0 (update to P using vector operations)
            best_sample = np.where(best_sample == 1,best_sample, 0)
            self.P = (1-learning_rate) * self.P + learning_rate * best_sample
            run_time = time.time() - start_time
            iteration_energies.append(sample_energies)
            iteration_times.append(run_time)
        return iteration_energies, iteration_times


def run_experiment(directory,file_prefix,N,seed_weights,iterations,population_size,learning_rate):
    pbil_model = PBIL(N,seed_weights)
    params = to_params(N,seed_weights,iterations,population_size,learning_rate)
    print('Running PBIL experiment with parameters:',params)
    start_time = time.time()
    iteration_energies, iteration_times = pbil_model.run(iterations,population_size,learning_rate)
    end_time = time.time()

    results = {
        'checkpoint': time.strftime("%Y_%m_%d_%H_%M_%S"),
        'iteration_times': iteration_times,
        'parameters': params,
        'iteration_energies': iteration_energies,
    }
    checkpoint = results['checkpoint']
    file =  join(directory,file_prefix)
    file = f'{file}_{checkpoint}_{start_time}_{end_time}.json'
    print('Creating file:',file)

    with open(file,'w') as results_file:
        dump(results,results_file)

def to_params(N,seed_weights,iterations,population_size,learning_rate):
    return {
        'N': N,
        'seed_weights': seed_weights,
        'iterations': iterations,
        'population_size': population_size,
        'learning_rate': learning_rate,
    }