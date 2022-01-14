import numpy as np
import cma
import time
from helper import init_weighted_max_sat_weights, init_weighted_max_sat_states
from json import dump
from os.path import join
from sys import maxsize

# Map continous states [-1,1]to {-1,1}
def normalise_sample(x):
    y = x.copy()
    y = np.where(y>0,1,y)
    y = np.where(y<=0,-1,y)
    return y

class CMAES:
    def __init__(self, N, seed_weights):
        self.N = N
        self.seed_weights = seed_weights
        self.weights = init_weighted_max_sat_weights(N,seed_weights)
    
    def cost(self, sample):
        sample = normalise_sample(sample)
        energy = -0.5 * (sample @ self.weights) @ sample.T
        return energy.flatten()[0]
    
    def run(self,x,sigma,iterations,population_size):
        options = {
            'bounds': [-1,1],
            'integer_variables': list(range(self.N)),
            'tolflatfitness': maxsize,
            'maxiter': iterations,
            'popsize': population_size,
            'timeout': '5*60*60',
            'verbose': 3,
        }
        es = cma.CMAEvolutionStrategy(x,sigma,options)
        iteration_energies = []
        iteration_times = []
        for i in range(iterations):
            start_time = time.time()
            samples = es.ask(number=population_size)
            sample_energies = [self.cost(sample) for sample in samples]
            es.tell(samples,sample_energies)
            run_time = time.time() - start_time
            iteration_energies.append(sample_energies)
            iteration_times.append(run_time)
        print(es.stop())
        return iteration_energies, iteration_times

def run_experiment(directory,file_prefix,N,seed_weights,seed_states,iterations,population_size,sigma):
    cmaes_model = CMAES(N,seed_weights)
    # Use the initial state configuration in self modelling as an initial point
    x0 = init_weighted_max_sat_states(N,seed_states)
    params = to_params(N,seed_weights,seed_states,iterations,population_size,sigma)
    print('Running CMA-ES experiment with parameters:',params)
    start_time = time.time()
    iteration_energies, iteration_times = cmaes_model.run(x0,sigma,iterations,population_size)
    end_time = time.time()

    results = {
        'checkpoint': time.strftime("%Y_%m_%d_%H_%M_%S"),
        'iteration_times': iteration_times,
        'parameters': params,
        'iteration_energies': iteration_energies,
    }
    checkpoint = results['checkpoint']
    file = join(directory,file_prefix)
    file = f'{file}_{checkpoint}_{start_time}_{end_time}.json'
    print('Creating file',file)

    with open(file,'w') as results_file:
        dump(results,results_file)

def to_params(N,seed_weights,seed_states,iterations,population_size,sigma):
    return {
        'N': N,
        'seed_weights': seed_weights,
        'seed_states': seed_states,
        'iterations': iterations,
        'population_size': population_size,
        'sigma': sigma,
    }