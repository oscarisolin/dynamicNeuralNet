"""Interactive plotting module for neural network training."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np


class InteractivePlot:
    """Manages the interactive matplotlib plot for training visualization."""
    
    def __init__(self, output_mapping, initial_iterations=1000):
        """Initialize the interactive plot.
        
        Args:
            output_mapping: Output mapping array
            initial_iterations: Initial number of iterations
        """
        self.output_mapping = output_mapping
        self.paused = [False]
        self.stopped = [False]
        self.update_freq = [10]
        self.next_iterations = [initial_iterations]
        self.autorounds = [initial_iterations]
        
        # Setup the figure
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = plt.subplot2grid((8, 5), (0, 0), colspan=5, rowspan=5)
        self.ax.set_title('Network Output vs Target During Training')
        self.ax.set_xlabel('Output Neuron Index')
        self.ax.set_ylabel('Value')
        self.ax.set_ylim([0, 1])
        self.ax.legend(['Target', 'Actual'])
        
        # Create control buttons
        self._setup_buttons()
        
        # Create slider
        self._setup_slider()
        
    def _setup_buttons(self):
        """Create and configure control buttons."""
        # Create button axes (row 6)
        ax_pause = plt.subplot2grid((8, 5), (6, 0))
        ax_stop = plt.subplot2grid((8, 5), (6, 1))
        ax_faster = plt.subplot2grid((8, 5), (6, 2))
        ax_slower = plt.subplot2grid((8, 5), (6, 3))
        ax_restart = plt.subplot2grid((8, 5), (6, 4))
        
        # Create buttons
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_faster = Button(ax_faster, 'Faster')
        self.btn_slower = Button(ax_slower, 'Slower')
        self.btn_restart = Button(ax_restart, 'Run More')
        
        # Connect button callbacks
        self.btn_pause.on_clicked(self._toggle_pause)
        self.btn_stop.on_clicked(self._stop_training)
        self.btn_faster.on_clicked(self._make_faster)
        self.btn_slower.on_clicked(self._make_slower)
        self.btn_restart.on_clicked(self._restart_training)
        
    def _setup_slider(self):
        """Create and configure the iteration slider."""
        # Create slider axis (row 7)
        ax_slider = plt.subplot2grid((8, 5), (7, 0), colspan=5)
        
        # Slider with logarithmic scale: 10^0 to 10^5 (1 to 100,000)
        self.slider_iterations = Slider(
            ax_slider, 'Iterations',
            0, 5, valinit=3, valstep=0.1,
            valfmt='%.0f iter'
        )
        
        self.slider_iterations.on_changed(self._update_slider_label)
        self._update_slider_label(3)  # Initialize label
        
    def _toggle_pause(self, event):
        """Toggle pause state."""
        self.paused[0] = not self.paused[0]
        self.btn_pause.label.set_text('Resume' if self.paused[0] else 'Pause')
        plt.draw()
    
    def _stop_training(self, event):
        """Stop training."""
        self.stopped[0] = True
    
    def _make_faster(self, event):
        """Increase plot update frequency."""
        self.update_freq[0] = max(1, self.update_freq[0] // 2)
        print(f'Update frequency: every {self.update_freq[0]} iterations')
    
    def _make_slower(self, event):
        """Decrease plot update frequency."""
        self.update_freq[0] = min(100, self.update_freq[0] * 2)
        print(f'Update frequency: every {self.update_freq[0]} iterations')
    
    def _restart_training(self, event):
        """Restart training with new iteration count."""
        self.autorounds[0] = self.next_iterations[0]
        self.stopped[0] = False
        self.paused[0] = False
        self.btn_pause.label.set_text('Pause')
        print(f'Starting {self.autorounds[0]} more iterations...')
        plt.draw()
    
    def _update_slider_label(self, val):
        """Update slider label with iteration count."""
        iterations = int(10 ** val)
        self.next_iterations[0] = iterations
        self.slider_iterations.valtext.set_text(f'{iterations} iter')
    
    def update_plot(self, zustand_t1, durchgang):
        """Update the plot with current network state.
        
        Args:
            zustand_t1: Current neuron states
            durchgang: Current iteration number
        """
        self.ax.clear()
        output_indices = self.output_mapping[:, 1].astype(int)
        target_values = self.output_mapping[:, 0]
        actual_values = zustand_t1[output_indices].flatten()
        x_pos = np.arange(len(output_indices))
        
        self.ax.bar(x_pos - 0.2, target_values, 0.4, 
                   label='Target', alpha=0.8, color='blue')
        self.ax.bar(x_pos + 0.2, actual_values, 0.4, 
                   label='Actual', alpha=0.8, color='orange')
        
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(output_indices)
        self.ax.set_xlabel('Output Neuron Index')
        self.ax.set_ylabel('Value')
        self.ax.set_ylim([0, 1])
        
        status = 'PAUSED' if self.paused[0] else 'RUNNING'
        self.ax.set_title(
            f'Network Training [{status}] - Iteration {durchgang} - '
            f'Update freq: {self.update_freq[0]}'
        )
        self.ax.legend()
        plt.pause(0.001)
    
    def handle_pause(self):
        """Handle pause state, blocking until resumed or stopped."""
        while self.paused[0] and not self.stopped[0]:
            plt.pause(0.1)
            if self.stopped[0]:
                self.autorounds[0] = 0
                plt.close(self.fig)
                return False
        return True
    
    def should_update_plot(self):
        """Check if plot should be updated this iteration."""
        return (self.autorounds[0] % self.update_freq[0]) == 0
    
    def is_stopped(self):
        """Check if training is stopped."""
        return self.stopped[0]
    
    def is_complete(self):
        """Check if all iterations are complete."""
        return self.autorounds[0] == 0
    
    def decrement_rounds(self):
        """Decrement the round counter."""
        if self.autorounds[0] > 0:
            self.autorounds[0] -= 1
    
    def close(self):
        """Close the plot window."""
        plt.close(self.fig)
    
    def print_completion_message(self):
        """Print message when training completes."""
        if self.is_stopped() and self.is_complete():
            print('Training stopped by user!')
        elif self.is_complete() and not self.is_stopped():
            print(f'Training completed! Click "Run More" to continue '
                  f'(slider set to {self.next_iterations[0]} iterations)')
