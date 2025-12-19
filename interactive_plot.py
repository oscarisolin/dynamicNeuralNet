"""Interactive plotting module for neural network training."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np


class InteractivePlot:
    """Manages the interactive matplotlib plot for training visualization."""
    
    def __init__(self, output_mapping, initial_iterations=100):
        """Initialize the interactive plot.
        
        Args:
            output_mapping: Output mapping array
            initial_iterations: Initial number of iterations
        """
        self.output_mapping = output_mapping
        self.paused = [False]
        self.stopped = [False]
        self.update_freq = [50]
        self.next_iterations = [initial_iterations]
        self.autorounds = [initial_iterations]

        # Neuron deletion control (interval in iterations)
        self.delete_interval = [200]

        # Growth control: only grow if error above this threshold
        self.grow_error_threshold = [0.02]
        
        # History tracking for error and network size
        self.iteration_history = []
        self.error_history = []
        self.size_history = []
        
        # Error tolerance for rollback (as percentage)
        self.error_tolerance = [10.0]  # percentage increase allowed
        
        # Setup the figure
        plt.ion()
        # Slightly taller figure to give more room for sliders
        self.fig = plt.figure(figsize=(14, 11))
        
        # Main plot for outputs (rows 0-3)
        self.ax = plt.subplot2grid((10, 5), (0, 0), colspan=5, rowspan=4)
        self.ax.set_title('Network Output vs Target During Training')
        self.ax.set_xlabel('Output Neuron Index')
        self.ax.set_ylabel('Value')
        self.ax.set_ylim([0, 1])
        self.ax.legend(['Target', 'Actual'])
        
        # Line plot for error and network size (rows 4-5)
        self.ax_metrics = plt.subplot2grid((10, 5), (4, 0), colspan=5, rowspan=2)
        self.ax_metrics.set_xlabel('Iteration')
        self.ax_metrics.set_ylabel('Error', color='red')
        self.ax_metrics.tick_params(axis='y', labelcolor='red')
        self.ax_metrics_twin = self.ax_metrics.twinx()
        self.ax_metrics_twin.set_ylabel('Network Size', color='blue')
        self.ax_metrics_twin.tick_params(axis='y', labelcolor='blue')
        
        # Create control buttons
        self._setup_buttons()
        
        # Create sliders
        self._setup_slider()

        # Adjust layout so widget titles don't overlap
        self.fig.subplots_adjust(top=0.95, bottom=0.06, hspace=0.9)
        
    def _setup_buttons(self):
        """Create and configure control buttons."""
        # Create button axes (row 7)
        ax_pause = plt.subplot2grid((10, 5), (7, 0))
        ax_stop = plt.subplot2grid((10, 5), (7, 1))
        ax_faster = plt.subplot2grid((10, 5), (7, 2))
        ax_slower = plt.subplot2grid((10, 5), (7, 3))
        ax_restart = plt.subplot2grid((10, 5), (7, 4))
        
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
        """Create and configure the sliders."""
        # Slider for neuron deletion interval (row 6, left side)
        # Use only columns 0-1 and leave column 2 empty as spacing
        ax_delete = plt.subplot2grid((10, 5), (6, 0), colspan=2)
        self.slider_delete_interval = Slider(
            ax_delete, 'Delete every (iter)',
            10, 1000, valinit=self.delete_interval[0], valstep=10,
            valfmt='%.0f'
        )
        self.slider_delete_interval.on_changed(self._update_delete_interval)
        self.slider_delete_interval.label.set_fontsize(8)
        self.slider_delete_interval.valtext.set_fontsize(8)

        # Slider for growth error threshold (row 6, right side), shifted
        # to columns 3-4 so there is a visual gap to the left slider
        ax_grow = plt.subplot2grid((10, 5), (6, 3), colspan=2)
        self.slider_grow_threshold = Slider(
            ax_grow, 'Grow if error >',
            0.0, 0.5, valinit=self.grow_error_threshold[0], valstep=0.001,
            valfmt='%.3f'
        )
        self.slider_grow_threshold.on_changed(self._update_grow_threshold)
        self.slider_grow_threshold.label.set_fontsize(8)
        self.slider_grow_threshold.valtext.set_fontsize(8)

        # Slider for iterations per run (row 8)
        ax_slider = plt.subplot2grid((10, 5), (8, 0), colspan=5)
        
        # Slider with logarithmic scale: 10^0 to 10^5 (1 to 100,000)
        self.slider_iterations = Slider(
            ax_slider, 'Iterations',
            0, 5, valinit=2, valstep=0.1,
            valfmt='%.0f iter'
        )
        
        self.slider_iterations.on_changed(self._update_slider_label)
        self._update_slider_label(2)  # Initialize label
        self.slider_iterations.label.set_fontsize(8)
        self.slider_iterations.valtext.set_fontsize(8)
        
        # Create error tolerance slider (row 9)
        ax_tolerance = plt.subplot2grid((10, 5), (9, 0), colspan=5)
        self.slider_tolerance = Slider(
            ax_tolerance, 'Error Tolerance (%)',
            1.0, 200.0, valinit=self.error_tolerance[0], valstep=1.0,
            valfmt='%.1f%%'
        )
        self.slider_tolerance.on_changed(self._update_tolerance)
        self.slider_tolerance.label.set_fontsize(8)
        self.slider_tolerance.valtext.set_fontsize(8)
        
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
    
    def _update_tolerance(self, val):
        """Update error tolerance value."""
        self.error_tolerance[0] = val

    def _update_delete_interval(self, val):
        """Update neuron deletion interval (in iterations)."""
        self.delete_interval[0] = int(val)

    def _update_grow_threshold(self, val):
        """Update error threshold above which growth is allowed."""
        self.grow_error_threshold[0] = float(val)
    
    def update_plot(self, zustand_t1, durchgang, error=None, network_size=None):
        """Update the plot with current network state.
        
        Args:
            zustand_t1: Current neuron states
            durchgang: Current iteration number
            error: Current error value
            network_size: Current network size
        """
        # Update history
        if error is not None and network_size is not None:
            self.iteration_history.append(durchgang)
            self.error_history.append(error)
            self.size_history.append(network_size)
        
        # Clear and update output bar chart
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
        
        # Update error and size line plots
        if len(self.iteration_history) > 0:
            self.ax_metrics.clear()
            self.ax_metrics_twin.clear()
            
            # Plot error
            self.ax_metrics.plot(self.iteration_history, self.error_history, 'r-', linewidth=2, label='Error')
            self.ax_metrics.set_xlabel('Iteration')
            self.ax_metrics.set_ylabel('Error', color='red')
            self.ax_metrics.tick_params(axis='y', labelcolor='red')
            self.ax_metrics.grid(True, alpha=0.3)
            
            # Plot network size
            self.ax_metrics_twin.plot(self.iteration_history, self.size_history, 'b-', linewidth=2, label='Network Size')
            self.ax_metrics_twin.set_ylabel('Network Size (neurons)', color='blue')
            self.ax_metrics_twin.tick_params(axis='y', labelcolor='blue')
        
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
