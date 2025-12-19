"""Neural network core functions."""

import numpy as np


class NeuralNetwork:
    """Dynamic neural network with learning capabilities."""
    
    def __init__(self, netsize=20, shrinking_factor=0.80):
        """Initialize the neural network.
        
        Args:
            netsize: Initial number of neurons
            shrinking_factor: Learning rate modifier
        """
        self.netsize = netsize
        self.shrinking_factor = shrinking_factor
        self.durchgang = 0
        
        # Initialize neuron states
        self.zustand_t = 2 * np.random.random((netsize, 1)) - 1
        self.zustand_t1 = np.copy(self.zustand_t)
        self.neuro_aktivitaet = np.zeros(netsize)
        
        # Initialize synapse matrix
        self.synapsenMatrix = np.multiply(
            (netsize * np.random.random((netsize, 3))).astype(int),
            [1, 1, (0.1 / netsize)]
        )
        
        # Will be set externally
        self.input_mapping = None
        self.output_mapping = None
    
    def set_mappings(self, input_mapping, output_mapping):
        """Set input and output mappings.
        
        Args:
            input_mapping: Input value-to-neuron mapping
            output_mapping: Output value-to-neuron mapping
        """
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
    
    def add_synapse(self, von, nach, gewicht=None):
        """Add a synapse connection.
        
        Args:
            von: Source neuron index
            nach: Target neuron index
            gewicht: Connection weight (random if None)
        """
        if gewicht is None:
            gewicht = 2 * np.random.random() - 1
        self.synapsenMatrix = np.append(
            self.synapsenMatrix, [[von, nach, gewicht]], axis=0
        )
    
    def remove_synapse(self, index):
        """Remove a synapse by index.
        
        Args:
            index: Index of synapse to remove
        """
        self.synapsenMatrix = np.delete(self.synapsenMatrix, index, axis=0)
    
    def add_neuron(self):
        """Add a new neuron with random connections."""
        last_index = len(self.zustand_t)
        self.zustand_t = np.append(self.zustand_t, [[0]], axis=0)
        self.zustand_t1 = np.copy(self.zustand_t)
        self.netsize += 1
        self.neuro_aktivitaet = np.zeros(self.netsize)
        
        # Add random connections
        for i in range(4):
            self.add_synapse(np.random.randint(last_index), last_index, 0.01)
            self.add_synapse(last_index, np.random.randint(last_index), 0.01)
    
    def remove_neuron(self, ind):
        """Remove a neuron if it's not an input or output neuron.
        
        Args:
            ind: Index of neuron to remove
        """
        # Don't remove input or output neurons
        if any(c in self.output_mapping[:, 1] for c in (ind, self.netsize - 1)):
            return
        if any(d in self.input_mapping[:, 1] for d in (ind, self.netsize - 1)):
            return
        
        # Remove neuron state
        self.neuro_aktivitaet = np.delete(
            self.neuro_aktivitaet, ind - 1, axis=0
        ).copy()
        self.zustand_t = np.delete(self.zustand_t, ind - 1, axis=0).copy()
        self.zustand_t1 = np.copy(self.zustand_t)
        
        # Remove associated synapses
        weg = []
        for key, x in enumerate(self.synapsenMatrix):
            if x[0] == ind or x[1] == ind:
                if ind not in self.input_mapping[:, 1]:
                    if ind not in self.output_mapping[:, 1]:
                        weg.append([key])
        
        self.synapsenMatrix = np.delete(self.synapsenMatrix, weg, axis=0)
        
        # Update neuron indices in mappings
        for ikey, b in enumerate(self.output_mapping[:, 1] > ind):
            if b:
                self.output_mapping[ikey, 1] -= 1
        
        for ikey, b in enumerate(self.input_mapping[:, 1] > ind):
            if b:
                self.input_mapping[ikey, 1] -= 1
        
        # Update synapse indices
        for ikey, b in enumerate(self.synapsenMatrix[:, 0] > ind):
            if b:
                self.synapsenMatrix[ikey, 0] -= 1
        
        for ikey, b in enumerate(self.synapsenMatrix[:, 1] > ind):
            if b:
                self.synapsenMatrix[ikey, 1] -= 1
        
        self.netsize -= 1
        
        # Clean up any invalid synapses (safety check)
        self._remove_invalid_synapses()
    
    def _remove_invalid_synapses(self):
        """Remove synapses that reference non-existent neurons."""
        valid_synapses = []
        for row in self.synapsenMatrix:
            if (row[0] < len(self.zustand_t) and row[1] < len(self.zustand_t)):
                valid_synapses.append(row)
        self.synapsenMatrix = np.array(valid_synapses) if valid_synapses else np.empty((0, 3))
    
    def step(self):
        """Perform one training step."""
        # Apply inputs
        for inp in self.input_mapping:
            self.zustand_t[int(inp[1])] = inp[0]
        
        # Reset states
        self.neuro_aktivitaet = np.zeros(self.netsize)
        self.zustand_t1 = np.zeros((len(self.zustand_t), 1))
        
        # Forward propagation
        for row in self.synapsenMatrix:
            if row[1] not in self.input_mapping[:, 1]:
                # Check both source and target indices are valid
                if (row[0] <= len(self.zustand_t) - 1 and 
                    row[1] <= len(self.zustand_t1) - 1):
                    self.zustand_t1[int(row[1])] += (
                        self.zustand_t[int(row[0])] * row[2]
                    )
        
        # Apply activation function
        for ind, outOfSum in np.ndenumerate(self.zustand_t1):
            self.zustand_t1[ind] = 1 / (1 + np.exp(-outOfSum))
        
        # Backpropagation
        zustand_tziel = np.copy(self.zustand_t1)
        
        # Set target values
        for oup in self.output_mapping:
            zustand_tziel[int(oup[1])] = oup[0]
        
        # Update weights
        for laufindex in range(self.netsize):
            delta_zstd = (
                (zustand_tziel[laufindex] - self.zustand_t1[laufindex]) *
                (self.zustand_t[laufindex] * (1 - self.zustand_t[laufindex]))
            )
            
            rows = np.where(self.synapsenMatrix[:, 1] == laufindex)
            for zeile in rows[0]:
                if len(self.zustand_t) > self.synapsenMatrix[zeile][0]:
                    self.synapsenMatrix[zeile][2] += (
                        delta_zstd * 
                        self.zustand_t[int(self.synapsenMatrix[zeile][0])]
                    ) * self.shrinking_factor
                    self.neuro_aktivitaet[laufindex] += abs(
                        self.synapsenMatrix[zeile][2]
                    )
        
        # Update neuron states
        self.zustand_t = np.copy(self.zustand_t1) * self.shrinking_factor
        self.durchgang += 1
        
        # Calculate and print error (before pruning changes network size)
        error = np.linalg.norm(zustand_tziel - self.zustand_t1)
        print(f'error {error:7.3f} groesse {self.netsize:3} lauf {self.durchgang:10}')
        
        # Remove isolated neurons with no connections (after error calculation)
        self.prune_isolated_neurons()
        
        return error
    
    def prune_inactive_neurons(self, threshold=0.04, max_removals=5):
        """Remove neurons with low activity.
        
        Args:
            threshold: Activity threshold for removal
            max_removals: Maximum neurons to remove in one call
        """
        hits = np.argwhere(abs(self.neuro_aktivitaet) < threshold)[:, 0]
        if hits.size > 0:
            # Select neurons to remove and sort in descending order
            num_to_remove = min(max_removals, len(hits))
            to_remove = np.random.choice(hits, size=num_to_remove, replace=False)
            # Remove in reverse order to maintain valid indices
            for i in sorted(to_remove, reverse=True):
                self.remove_neuron(i)
    
    def prune_isolated_neurons(self):
        """Remove neurons that have no synapses connected to them at all."""
        # Get all neuron indices that appear in synapses
        connected_neurons = set()
        for synapse in self.synapsenMatrix:
            connected_neurons.add(int(synapse[0]))
            connected_neurons.add(int(synapse[1]))
        
        # Find neurons that have no connections
        isolated = []
        for i in range(self.netsize):
            if i not in connected_neurons:
                # Don't remove input or output neurons even if isolated
                if (i not in self.input_mapping[:, 1] and 
                    i not in self.output_mapping[:, 1]):
                    isolated.append(i)
        
        # Remove isolated neurons (in reverse order to maintain indices)
        for neuron_idx in sorted(isolated, reverse=True):
            self.remove_neuron(neuron_idx)
    
    def prune_weak_synapses(self, threshold=0.01, max_removals=4):
        """Remove synapses with weak connections.
        
        Args:
            threshold: Weight threshold for removal
            max_removals: Maximum synapses to remove
        """
        weak_synapses = np.argwhere(abs(self.synapsenMatrix[:, 2]) < threshold)[:, 0]
        # Remove in reverse order to maintain valid indices
        for i in sorted(weak_synapses[:max_removals], reverse=True):
            self.remove_synapse(i)
    
    def grow_network(self):
        """Add neurons to highly active areas."""
        high_activity = np.argwhere(abs(self.neuro_aktivitaet) > 10)
        if len(high_activity) > 2:
            self.add_synapse(
                np.random.choice(high_activity[:, 0]),
                np.random.choice(high_activity[:, 0]),
                0.01
            )
    
    def print_state(self):
        """Print current network state."""
        print("zustand: \n {} \n".format(self.zustand_t1))
        print("laenge zustand: \n {} \n".format(len(self.zustand_t1)))
    
    def print_synapses(self):
        """Print synapse matrix."""
        for d in self.synapsenMatrix:
            print(f"{int(d[0])},\t{int(d[1])},\t {d[2]}")
        print("laenge synapsenMatrix: \n {} \n".format(len(self.synapsenMatrix)))
    
    def print_output(self):
        """Print current output values."""
        print('only output \n index \n {} \n loesung: \n {} \n'.format(
            self.output_mapping[:, 1],
            self.zustand_t1[self.output_mapping[:, 1].astype(int)]
        ))
    
    def print_activity(self):
        """Print neuron activity."""
        print("neuro aktivitaet: \n {} \n".format(self.neuro_aktivitaet))
        print("laenge neuro aktivit.: \n {} \n".format(len(self.neuro_aktivitaet)))
