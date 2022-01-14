import numpy as np
import time
import json
from helper import init_weighted_max_sat_states, init_weighted_max_sat_weights
from os.path import join


class HopfieldNetworkRC:
    def __init__(self,N,seed_weights,seed_states):
        self.N = N
        self.seed_weights = seed_weights
        self.seed_states = seed_states
        self.initial_weights = init_weighted_max_sat_weights(N,seed_weights)
        self.weights = self.initial_weights.copy()
        self.states = init_weighted_max_sat_states(N,seed_states)
    
    def reset_states(self):
        return np.random.choice([-1, +1], self.N)
    
    def get_energy(self):
        energy = - 0.5 * (self.states @ self.initial_weights) @ self.states.T
        return energy
    
    def update_async(self, neuron_idx):
        value = self.states @ self.weights[:,neuron_idx]
        self.states[neuron_idx] = 1 if value > 0 else -1
    
    def update_weights(self, learning_rate):
        self.weights = self.weights + learning_rate * (self.states.reshape((-1, 1)) * self.states.reshape((1, -1)) - np.eye(self.N))
        
    # hebbian_learning_relaxation: The relaxation to start applying hebbian learning
    def run(self,relaxations,state_updates,hebbian_learning_relaxation,learning_rate):
        relaxation_energies = []
        relaxation_times = []
        hebbian_learning = False
        for r in range(1,relaxations+1):
            if r == hebbian_learning_relaxation:
                hebbian_learning = True
            self.states = self.reset_states()
            times = []
            energies = [] 
            for s in range(state_updates):
                start_time = time.time()
                neuron_idx = np.random.randint(self.N)
                self.update_async(neuron_idx)
                if hebbian_learning:
                    self.update_weights(learning_rate)
                run_time = time.time() - start_time
                energies.append(self.get_energy())
                times.append(run_time)
            relaxation_times.append(times)
            relaxation_energies.append(energies)
        return relaxation_energies, relaxation_times
    
    #def run(self,relaxations,state_updates,hebbian_learning,learning_rate):
        #relaxation_energies = []
        #relaxation_times = []
        #for r in range(relaxations):
            ## Reset states
            #self.states = self.reset_states()
            #times = []
            #energies = []
            #for s in range(state_updates):
                #start_time = time.time()
                ## Update states randomly or deterministically
                #neuron_idx = np.random.randint(self.N)
                ##neuron_idx = np.random.randint(self.N) if random else i % self.N
                #self.update_async(neuron_idx)
                #if hebbian_learning:
                    #self.update_weights(learning_rate)
                #run_time = time.time() - start_time
                #energies.append(self.get_energy())
                #times.append(run_time)
            #relaxation_times.append(times)
            #relaxation_energies.append(energies)
        #return relaxation_energies, relaxation_times

def run_experiment(directory,file_prefix, N,seed_weights,seed_states,relaxations,state_updates,hebbian_learning_relaxation,learning_rate):
    self_modelling_model = HopfieldNetworkRC(N,seed_weights,seed_states)
    params = to_params( N = N, seed_weights=seed_weights, seed_states=seed_states, relaxations=relaxations, state_updates=state_updates, hebbian_learning_relaxation=hebbian_learning_relaxation, learning_rate=learning_rate)
    print('Running self-modelling experiment with parameters:',params)

    start_time = time.time()
    relaxation_energies, relaxation_times = self_modelling_model.run( relaxations=relaxations, state_updates=state_updates, hebbian_learning_relaxation=hebbian_learning_relaxation, learning_rate=learning_rate)
    end_time = time.time()
    
    results = {
        'checkpoint': time.strftime("%Y_%m_%d_%H_%M_%S"),
        'relaxation_times': relaxation_times,
        'parameters': params,
        'relaxation_energies': relaxation_energies,
    }
    checkpoint = results['checkpoint']
    file = join(directory,file_prefix)
    file = f'{file}_{checkpoint}_{start_time}_{end_time}.json'

    print('Creating file...',file)
    with open(file,'w') as results_file:
        json.dump(results,results_file)

def to_params(N,seed_weights,seed_states,relaxations,state_updates,hebbian_learning_relaxation,learning_rate):
    return {
        'N': N,
        'seed_weights': seed_weights,
        'seed_states': seed_states,
        'relaxations': relaxations,
        'state_updates': state_updates,
        'hebbian_learning_relaxation': hebbian_learning_relaxation,
        'learning_rate': learning_rate,
    }