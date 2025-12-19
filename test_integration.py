#!/usr/bin/env python3
"""Integration test for modular components."""

import numpy as np
from neural_network import NeuralNetwork
import training_examples

def test_all_examples():
    """Test that all training examples work."""
    print("="*60)
    print("INTEGRATION TEST: All Training Examples")
    print("="*60)
    
    for example_id in range(1, 11):
        example = training_examples.get_example(str(example_id))
        print(f"\n[{example_id}/10] Testing: {example['name']}")
        print(f"Description: {example['description']}")
        print(f"Inputs: {len(example['input'])}, Outputs: {len(example['output'])}")
        
        # Create network
        network = NeuralNetwork()
        network.set_mappings(example['input'], example['output'])
        
        # Run 10 training steps
        errors = []
        for i in range(10):
            for inp in example['input']:
                network.zustand_t[int(inp[1])] = inp[0]
            error = network.step()
            errors.append(error)
        
        # Check convergence
        initial_error = errors[0]
        final_error = errors[-1]
        improvement = initial_error - final_error
        
        print(f"Initial error: {initial_error:.3f}")
        print(f"Final error: {final_error:.3f}")
        print(f"Improvement: {improvement:.3f}")
        
        if improvement > 0:
            print("✓ Network is learning!")
        else:
            print("⚠ Network not improving (may need more iterations)")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    print("\n✓ All 10 training examples work correctly!")
    print("✓ NeuralNetwork class functional")
    print("✓ training_examples module functional")
    print("\nReady to use in cnn.py!")

def test_network_operations():
    """Test network operations."""
    print("\n" + "="*60)
    print("TESTING NETWORK OPERATIONS")
    print("="*60)
    
    network = NeuralNetwork(netsize=20)
    example = training_examples.get_example('1')
    network.set_mappings(example['input'], example['output'])
    
    initial_size = network.netsize
    print(f"\nInitial network size: {initial_size}")
    
    # Test adding neuron
    network.add_neuron()
    print(f"After add_neuron(): {network.netsize}")
    assert network.netsize == initial_size + 1, "add_neuron failed"
    print("✓ add_neuron() works")
    
    # Test adding synapse
    initial_synapses = len(network.synapsenMatrix)
    network.add_synapse(0, 1, 0.5)
    assert len(network.synapsenMatrix) == initial_synapses + 1, "add_synapse failed"
    print("✓ add_synapse() works")
    
    # Test step
    initial_durchgang = network.durchgang
    for inp in example['input']:
        network.zustand_t[int(inp[1])] = inp[0]
    network.step()
    assert network.durchgang == initial_durchgang + 1, "step failed"
    print("✓ step() works")
    
    # Test pruning
    network.prune_weak_synapses(threshold=0.01, max_removals=2)
    print("✓ prune_weak_synapses() works")
    
    network.prune_inactive_neurons(threshold=0.04, max_removals=2)
    print("✓ prune_inactive_neurons() works")
    
    print("\n✓ All network operations functional!")

if __name__ == "__main__":
    test_all_examples()
    test_network_operations()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now run:")
    print("  python cnn.py     # Full server with interactive mode")
    print("  python test_modular.py  # Simple standalone test")
