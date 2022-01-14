import numpy as np

class HopfieldNetworkRC:
    """ Class for the Hopfield Network with random problem structure.
    
    Attributes
    ----------
    neurons : int 
        The number of neurons in the network.
    states : numpy.ndarray
        Network state matrix of shape (1, neurons).
    weights : numpy.ndarray
        Weight matrix of shape (neurons, neurons). With the application of Hebbian learning, 
        these weights will be modified.
    initial_weights : numpy.ndarray
        Weight matrix of shape (neurons, neurons). Represents original weights of the network.
    """
    def __init__(self, n):
        self.neurons = n
        self.states = self.init_states()
        self.weights = self.init_weights()
        self.initial_weights = self.weights.copy()
    
    def init_states(self):
        """Initialises a random bipolar network configuration.
        
        Returns
        -------
        numpy.ndarray
            Network states of shape (1, self.neurons)
        """
        
        return np.random.choice([-1, +1], self.neurons)
    
    def init_weights(self):
        """Initialises the connection weights of the network (symmetric, diagonal 0, containing bipolar values)
        
        Returns
        -------
        weights: numpy.ndarray
            Connection weights of shape (self.neurons, self.neurons)
        """
        weights = np.random.choice([-1, +1], size = (self.neurons, self.neurons))
        #weights = np.random.uniform(-1, 1, size = (self.neurons, self.neurons))
        weights = weights - np.diag(np.diag(weights))
        weights = np.tril(weights) + np.tril(weights, -1).T
        return weights
    
    def get_energy(self):
        """Calculates the energy of the network based on the initial/original weights of the network.
        
        Returns
        -------
        energy: float
        """
        energy = - 0.5 * (self.states @ self.initial_weights) @ self.states.T
        return energy
    
    def update_async(self, neuron_idx):
        """Updates the state of the chosen neuron.
        
        Parameters
        ----------
        neuron_idx: int
            index of the neuron to update
            
        Returns
        -------
        None
        """
        value = self.states @ self.weights[:,neuron_idx]
        self.states[neuron_idx] = 1 if value > 0 else -1
    
    def update_weights(self, learning_rate):
        """Applies hebbian learning and modifies the weights of the network.
        Note that initial/original weights of the network are not modified!
        
        Parameters
        ----------
        learning_rate: float
            The learning rate should be small enough such that the weights of the network
            are modified slowly.
        
        Returns
        -------
        None
        """
        self.weights = self.weights + learning_rate * (self.states.reshape((-1, 1)) * self.states.reshape((1, -1)) - np.eye(self.neurons))
        
    def run_random_restarts(self, relaxations=1000, state_updates=1000, random=True, hebbian_learning=False, learning_rate=0.000001):
        """Runs multiple relaxaions and state updates.
        
        Parameters
        ----------
        relaxations: int
            The number of times resetting the network with a random state configuration.
        state_updates: int
            The number of times updating the state of the network.
        random: bool
            Randomly select nodes for update or not.
            There are two ways of updating the network: randomly select a node each time or 
            update nodes deterministically, ensuring all nodes are selected (select node 0, node 1,... and again).
        hebbian_learning: bool
            Apply Hebbian learning to the connection weights.
        learning_rate: float
            Learning rate for the Hebbian learning.
        
        Returns
        -------
        relaxation_energies: list
            Sampled energies at each time step ([[from relaxation 1], [from relaxation 2], ...])
        """
        relaxation_energies = []
        
        for _ in range(relaxations):
            # Reset states
            self.states = self.init_states()
            energies = []
            for i in range(state_updates):
                # Update states randomly or deterministically
                neuron_idx = np.random.randint(self.neurons) if random else i % self.neurons
                self.update_async(neuron_idx)
                if hebbian_learning:
                    self.update_weights(learning_rate)
                energies.append(self.get_energy())
            relaxation_energies.append(energies)
            
        return relaxation_energies