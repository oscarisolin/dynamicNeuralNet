"""Test neural network with interactive live plotting using modular components."""

import numpy as np
from neural_network import NeuralNetwork
from interactive_plot import InteractivePlot

# Setup training example
input_mit_mapping = np.array([[0.5, 0], [1, 1], [0, 3], [0, 4]])
output_mit_mapping = np.array([
    [0.321, 5],
    [0.9, 6],
])

# Test the interactive plotting with slider and restart
print("Testing neural network with advanced interactive controls...")
print("Running training with live plot updates")
print("\nControls:")
print("  - Click 'Pause' to pause/resume training")
print("  - Click 'Stop' to end training permanently")
print("  - Click 'Faster' to update plot more frequently")
print("  - Click 'Slower' to update plot less frequently")
print("  - Click 'Run More' to run additional iterations")
print("  - Use SLIDER to set how many iterations to run next\n")

# Initialize network
network = NeuralNetwork()
network.set_mappings(input_mit_mapping, output_mit_mapping)

# Create interactive plot
plot = InteractivePlot(output_mit_mapping, initial_iterations=100)

def make_faster(event):
    plot.update_freq[0] = max(1, plot.update_freq[0] // 2)
    print(f'Update frequency: every {plot.update_freq[0]} iterations')

def make_slower(event):
    plot.update_freq[0] = min(50, plot.update_freq[0] * 2)
    print(f'Update frequency: every {plot.update_freq[0]} iterations')

def restart_training(event):
    plot.autorounds[0] += plot.next_iterations[0]
    plot.stopped[0] = False
    plot.paused[0] = False
    plot.btn_pause.label.set_text('Pause')
    print(f'Running {plot.next_iterations[0]} more iterations...')

# Override button callbacks to use custom functions
plot.btn_faster.on_clicked(make_faster)
plot.btn_slower.on_clicked(make_slower)
plot.btn_restart.on_clicked(restart_training)

print("Starting training loop...")
print("When it completes, adjust the slider and click 'Run More' to continue!\n")

# Training loop
while plot.autorounds[0] > 0 or not plot.is_stopped():
    # Check if we should wait for user to click "Run More"
    if plot.autorounds[0] == 0 and not plot.is_stopped():
        plot.print_completion_message()
        # Wait for user interaction
        while plot.autorounds[0] == 0 and not plot.is_stopped():
            import matplotlib.pyplot as plt
            plt.pause(0.1)
        if plot.is_stopped():
            break
    
    # Check if stopped
    if plot.is_stopped():
        print(f'\nTraining stopped at iteration {network.durchgang}')
        break
    
    # Pause handling
    plot.handle_pause()
    if plot.is_stopped():
        break
    
    if not plot.is_stopped() and plot.autorounds[0] > 0:
        # Apply inputs
        for inp in input_mit_mapping:
            network.zustand_t[int(inp[1])] = inp[0]
        
        # Train
        network.step()
        
        # Update plot
        if plot.should_update_plot():
            plot.update_plot(network.zustand_t1, network.durchgang)
        
        # Decrement rounds
        plot.decrement_rounds()

print("\n\nTraining complete!")
network.print_output()

plot.close()
