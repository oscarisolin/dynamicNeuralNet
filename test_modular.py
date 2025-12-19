#!/usr/bin/env python3
"""Test the modularized neural network."""

import numpy as np
from neural_network import NeuralNetwork
from interactive_plot import InteractivePlot
import training_examples

def test_basic_training():
    """Test basic training loop."""
    # Get a training example
    example = training_examples.get_example('2')  # XOR gate
    print(f"Testing: {example['name']}")
    print(f"Description: {example['description']}")
    
    # Create network
    network = NeuralNetwork()
    network.set_mappings(example['input'], example['output'])
    
    # Create plot
    plot = InteractivePlot(example['output'], initial_iterations=100)
    
    # Training loop
    while plot.autorounds[0] > 0 and not plot.is_stopped():
        # Apply inputs
        for inp in example['input']:
            network.zustand_t[int(inp[1])] = inp[0]
        
        # Train
        error = network.step()
        
        # Update plot
        if plot.should_update_plot():
            plot.update_plot(network.zustand_t1, network.durchgang)
        
        # Decrement autorounds
        plot.decrement_rounds()
        
        # Handle pause
        plot.handle_pause()
    
    if plot.is_stopped():
        print("Training stopped by user")
    else:
        print("Training completed!")
        network.print_output()
    
    plot.close()

if __name__ == "__main__":
    test_basic_training()
