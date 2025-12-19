# Dynamic Neural Network - Interactive Training with Live Visualization

A modular, self-organizing neural network implementation with real-time visualization and interactive controls. Features 10 preset training examples and dynamic network structure that grows and prunes during training.

## 🚀 Quick Start

### Option 1: Full Interactive Mode (Recommended)
```bash
python cnn.py
```
Enter `ll` when prompted, then select a training example (1-10).

### Option 2: Quick Test
```bash
python test_modular.py
```

### Option 3: Integration Test
```bash
python test_integration.py
```

## 📁 Project Structure

```
.
├── cnn.py                    # Main program with websocket server
├── neural_network.py         # NeuralNetwork class - core network logic
├── interactive_plot.py       # InteractivePlot class - visualization
├── training_examples.py      # 10 preset training examples
├── test_modular.py          # Simple test script
├── test_integration.py      # Full integration test
└── visual.html              # D3.js visualization frontend
```

## 🧠 Core Modules

### `neural_network.py` - NeuralNetwork Class

Encapsulates all network operations with a clean API:

**Initialization:**
```python
network = NeuralNetwork(netsize=20, shrinking_factor=0.80)
network.set_mappings(input_mapping, output_mapping)
```

**Key Methods:**
- `step()` - Perform one training iteration with backpropagation
- `add_neuron()` / `remove_neuron(ind)` - Dynamic structure modification
- `add_synapse(from, to, weight)` / `remove_synapse(index)` - Connection management
- `prune_inactive_neurons(threshold, max_removals)` - Remove low-activity neurons
- `prune_weak_synapses(threshold, max_removals)` - Remove weak connections
- `grow_network()` - Add synapses to highly active areas
- `print_output()`, `print_state()`, `print_synapses()`, `print_activity()` - Debug output

**Automatic Management:**
- Forward propagation through synapse matrix
- Sigmoid activation function
- Backpropagation with configurable learning rate
- Error calculation and reporting

### `interactive_plot.py` - InteractivePlot Class

Real-time matplotlib visualization with interactive controls:

**Features:**
- **Live Bar Chart**: Blue bars (target) vs Orange bars (actual outputs)
- **Interactive Buttons**:
  - **Pause/Resume** - Pause/resume training
  - **Stop** - Stop training and close plot
  - **Faster** - Update visualization more frequently
  - **Slower** - Update visualization less frequently
  - **Run More** - Start additional iterations
- **Logarithmic Slider**: Set iterations from 1 to 100,000

**Usage:**
```python
plot = InteractivePlot(output_mapping, initial_iterations=1000)

while plot.autorounds[0] > 0 and not plot.is_stopped():
    # Training code...
    if plot.should_update_plot():
        plot.update_plot(network.zustand_t1, network.durchgang)
    plot.decrement_rounds()
    plot.handle_pause()

plot.close()
```

**Visual Layout:**
- 14×8 inch figure window
- 8×5 subplot grid
- Main plot: rows 0-4, all columns
- Buttons: row 6, 5 columns
- Slider: row 7, spans all columns

### `training_examples.py` - Training Examples Library

10 diverse preset training examples:

| # | Name | Description | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 1 | Simple Binary | Learn two output values | 4 | 2 |
| 2 | XOR Gate | Classic XOR problem | 4 | 2 |
| 3 | AND Gate | Logical AND operation | 4 | 1 |
| 4 | Multi-Target | Multiple output patterns | 4 | 6 |
| 5 | Linear Pattern | Linear relationships | 4 | 3 |
| 6 | Symmetric Pattern | Symmetric outputs | 4 | 5 |
| 7 | Binary Counter | Counter sequence | 4 | 4 |
| 8 | High Precision | 9 precise decimal targets | 3 | 9 |
| 9 | Sparse Pattern | Few inputs, many outputs | 2 | 5 |
| 10 | Random Challenge | Random target values | 4 | 4 |

**API:**
```python
import training_examples

# List all examples
print(training_examples.list_examples())

# Get specific example
example = training_examples.get_example('2')  # XOR Gate
print(example['name'])
print(example['description'])
input_mapping = example['input']
output_mapping = example['output']
```

**Adding Custom Examples:**
Edit `training_examples.py` and add to the `TRAINING_EXAMPLES` dictionary:
```python
'11': {
    'name': 'My Custom Example',
    'description': 'What it does',
    'input': np.array([[value, neuron_index], ...]),
    'output': np.array([[value, neuron_index], ...])
}
```

## 🎮 Interactive Training Mode

### Starting Training

1. Run `python cnn.py`
2. Enter `ll` for interactive training
3. Select a training example:

```
==================================================
TRAINING EXAMPLE SELECTION
==================================================

=== Available Training Examples ===

1. Simple Binary (2 outputs)
   Learn two simple output values
   Inputs: 4, Outputs: 2

2. XOR Gate
   Classic XOR problem
   Inputs: 4, Outputs: 2

... (examples 3-10) ...

Or press Enter to use current custom mapping

Select example (1-10) or press Enter:
```

### During Training

**Console Output:**
```
error   0.712 groesse  20 lauf          1
error   0.571 groesse  20 lauf        100
```
- **error**: Distance from targets (lower is better)
- **groesse**: Network size (number of neurons)
- **lauf**: Iteration count

**Interactive Controls:**
- **Pause/Resume Button**: Pause/resume without losing state
- **Stop Button**: Terminate and close
- **Faster/Slower Buttons**: Adjust visualization update frequency (doesn't affect training speed!)
- **Run More Button**: Continue training with slider-specified iterations
- **Slider**: Set next iteration count (1 to 100,000, logarithmic scale)

**Tips:**
- Click "Slower" for faster training (updates plot less frequently)
- Training continues at full speed while paused (just blocks until resumed)
- Use slider before clicking "Run More" to continue training

## 📋 All Commands (cnn.py)

When running `python cnn.py`, available commands:

| Command | Description |
|---------|-------------|
| `ll` | Interactive training with example selection |
| `t` | Train one step |
| `a` | Add a neuron |
| `d` | Delete weak synapses |
| `s` | Forward pass only (no training) |
| `p` | Print synapse matrix |
| `pz` | Print neuron states |
| `op` | Print output mapping |
| `ip` | Print input mapping |
| `na` | Print neuron activity |
| `e` | Exit program |

## 🔧 How It Works

### Network Architecture

**Dynamic Structure:**
- Starts with 20 neurons
- Automatically grows during training (adds neurons to active areas)
- Automatically prunes during training (removes inactive neurons and weak synapses)

**Training Schedule:**
- Every 20 iterations: Prune up to 5 inactive neurons (activity < 0.04)
- Every 100 iterations: Prune up to 4 weak synapses (weight < 0.01)
- Every 60 iterations (before iteration 3000): Add neuron to growing areas
- Every 70 iterations (before iteration 3000): Add synapse between active neurons

**Learning Algorithm:**
1. Set input neuron values
2. Forward propagation through synapse matrix
3. Sigmoid activation: `1 / (1 + exp(-sum))`
4. Calculate error: `target - actual`
5. Backpropagation: Update weights using gradient descent
6. Apply shrinking factor (learning rate): 0.80

### Mapping Format

**Input/Output mappings** are NumPy arrays with format: `[[value, neuron_index], ...]`

Example:
```python
input_mapping = np.array([
    [0.5, 0],   # Set neuron 0 to 0.5
    [1.0, 1],   # Set neuron 1 to 1.0
    [0.0, 3],   # Set neuron 3 to 0.0
])

output_mapping = np.array([
    [0.321, 5], # Target for neuron 5: 0.321
    [0.900, 6], # Target for neuron 6: 0.900
])
```

## 💡 Usage Examples

### Example 1: Basic Training Loop

```python
from neural_network import NeuralNetwork
from interactive_plot import InteractivePlot
import training_examples

# Get example
example = training_examples.get_example('2')  # XOR gate

# Create network
network = NeuralNetwork()
network.set_mappings(example['input'], example['output'])

# Create plot
plot = InteractivePlot(example['output'], initial_iterations=1000)

# Training loop
while plot.autorounds[0] > 0 and not plot.is_stopped():
    # Apply inputs
    for inp in example['input']:
        network.zustand_t[int(inp[1])] = inp[0]
    
    # Train
    network.step()
    
    # Update visualization
    if plot.should_update_plot():
        plot.update_plot(network.zustand_t1, network.durchgang)
    
    plot.decrement_rounds()
    plot.handle_pause()

plot.close()
```

### Example 2: Training Without Visualization

```python
from neural_network import NeuralNetwork
import training_examples

example = training_examples.get_example('3')  # AND gate
network = NeuralNetwork(netsize=20, shrinking_factor=0.80)
network.set_mappings(example['input'], example['output'])

# Train for 1000 iterations
for i in range(1000):
    for inp in example['input']:
        network.zustand_t[int(inp[1])] = inp[0]
    error = network.step()
    
    if i % 100 == 0:
        print(f"Iteration {i}: Error = {error:.3f}")

# Print final results
network.print_output()
```

### Example 3: Custom Training Example

```python
import numpy as np
from neural_network import NeuralNetwork

# Define custom mappings
custom_input = np.array([[0.7, 0], [0.3, 1]])
custom_output = np.array([[0.5, 5], [0.8, 6], [0.2, 7]])

# Train
network = NeuralNetwork()
network.set_mappings(custom_input, custom_output)

for i in range(500):
    for inp in custom_input:
        network.zustand_t[int(inp[1])] = inp[0]
    network.step()
```

## 🌐 Websocket Server Mode

The main program includes a websocket server for real-time visualization with D3.js:

**Server Details:**
- Host: `0.0.0.0`
- Port: `5678`
- Sends neuron activity and synapse data in real-time
- Frontend: [visual.html](visual.html) for browser-based visualization

**Data Format:**
```json
// Message 1: Network state
[
    [activity_values...],      // Neuron activities
    [[from, to, weight], ...]  // Synapse connections
]

// Message 2: Training data
{
    "input": [[value, neuron], ...],
    "output": [[value, neuron], ...]
}
```

## 📊 Recommended Training Examples

### For Beginners
- **Example 1 (Simple Binary)**: Easiest, 2 outputs, converges quickly
- **Example 3 (AND Gate)**: Single output, very straightforward

### Classic Problems
- **Example 2 (XOR Gate)**: The classic neural network challenge
- **Example 7 (Binary Counter)**: Sequential pattern learning

### Advanced Challenges
- **Example 4 (Multi-Target)**: 6 outputs, requires more iterations
- **Example 8 (High Precision)**: 9 outputs with precise decimal values
- **Example 10 (Random Challenge)**: Hardest - random target values

## 📈 Success Metrics

**Good convergence indicators:**
- Error drops below 0.1
- Orange bars align with blue bars in visualization
- Network size stabilizes (stops growing/shrinking dramatically)
- Consistent low error across iterations

**If network isn't learning:**
- Use slider to set 10,000+ iterations
- Try a different example (some are harder than others)
- Check that mappings make sense (valid neuron indices)
- XOR and other non-linear problems need more time

## 🔬 Architecture Benefits

### Before Modularization
- ~500 lines of monolithic code
- Global variables everywhere
- Mixed concerns (plotting + network + server)
- Hard to test individual components
- Difficult to add new features

### After Modularization
- Clean separation of concerns
- Reusable components
- Easy to test (see test_modular.py, test_integration.py)
- Simple to add new training examples
- Clear, documented API
- ~200 lines for main program

## 🛠️ Dependencies

```bash
pip install numpy matplotlib websockets
```

**Required:**
- `numpy` - Array operations and neural network computations
- `matplotlib` - Interactive plotting and visualization

**Optional:**
- `websockets` - Only needed for server mode (cnn.py)

## 🧪 Testing

### Run All Tests
```bash
# Simple standalone test (XOR example, 100 iterations)
python test_modular.py

# Full integration test (all 10 examples + operations)
python test_integration.py
```

### Test Results
All tests passing:
- ✓ All 10 training examples functional
- ✓ Network operations work (add/remove neurons/synapses)
- ✓ Pruning and growth mechanisms functional
- ✓ Interactive plotting with all controls
- ✓ 9/10 examples show learning improvement in first 10 iterations

## 🐛 Troubleshooting

**Plot window doesn't appear:**
- Ensure matplotlib is installed
- Check you're in a graphical environment (not SSH without X forwarding)
- Try: `export DISPLAY=:0` on Linux

**Import errors:**
- Install dependencies: `pip install numpy matplotlib websockets`
- Activate virtual environment if using one

**Network not learning:**
- Increase iterations (10,000+ for complex problems like XOR)
- Verify input/output mappings use valid neuron indices
- Check that target values are between 0 and 1
- Some problems inherently need more time

**"Network not initialized" errors:**
- Run `ll` mode first to initialize the network
- Or use standalone test scripts

**Training too slow:**
- Click "Slower" button (counterintuitive but updates plot less often)
- Visualization updates slow down training, not the other way around
- Training happens at full speed regardless of plot frequency

## 🎯 Advanced Customization

### Modify Network Parameters
```python
network = NeuralNetwork(
    netsize=40,              # Start with more neurons
    shrinking_factor=0.90    # Higher learning rate
)
```

### Adjust Pruning/Growth Schedules
Edit [neural_network.py](neural_network.py) `step()` method thresholds or frequencies.

### Change Visualization Settings
```python
plot = InteractivePlot(
    output_mapping,
    initial_iterations=5000  # Start with more iterations
)

# Access internal settings
plot.update_freq[0] = 5  # Update every 5 iterations
```

### Add Activation Functions
Modify `step()` in [neural_network.py](neural_network.py):
```python
# Current: Sigmoid
self.zustand_t1[ind] = 1 / (1 + np.exp(-outOfSum))

# Alternative: ReLU
self.zustand_t1[ind] = max(0, outOfSum)

# Alternative: Tanh
self.zustand_t1[ind] = np.tanh(outOfSum)
```

## 📝 Code Example: Complete Training Session

```python
#!/usr/bin/env python3
"""Complete training example."""

from neural_network import NeuralNetwork
from interactive_plot import InteractivePlot
import training_examples

def train_with_visualization():
    """Train network with live visualization."""
    # Select training example
    print(training_examples.list_examples())
    choice = input("Select example (1-10): ")
    example = training_examples.get_example(choice)
    
    print(f"\nTraining: {example['name']}")
    print(f"{example['description']}\n")
    
    # Initialize
    network = NeuralNetwork(netsize=20, shrinking_factor=0.80)
    network.set_mappings(example['input'], example['output'])
    plot = InteractivePlot(example['output'], initial_iterations=1000)
    
    # Training loop
    iteration = 0
    while plot.autorounds[0] > 0 and not plot.is_stopped():
        # Apply inputs
        for inp in example['input']:
            network.zustand_t[int(inp[1])] = inp[0]
        
        # Train one step
        error = network.step()
        
        # Update visualization
        if plot.should_update_plot():
            plot.update_plot(network.zustand_t1, network.durchgang)
        
        # Manage rounds
        plot.decrement_rounds()
        
        # Handle pause
        plot.handle_pause()
        
        iteration += 1
    
    # Results
    if plot.is_stopped():
        print("\nTraining stopped by user")
    else:
        print("\nTraining completed!")
        network.print_output()
    
    plot.close()

if __name__ == "__main__":
    train_with_visualization()
```

## 📚 Further Reading

- Check the code comments in each module for detailed documentation
- [neural_network.py](neural_network.py) - Network implementation details
- [interactive_plot.py](interactive_plot.py) - Visualization internals
- [training_examples.py](training_examples.py) - Example definitions
- [test_modular.py](test_modular.py) - Simple usage example
- [test_integration.py](test_integration.py) - Comprehensive testing

## 🤝 Contributing

To add new training examples:
1. Edit `TRAINING_EXAMPLES` dict in [training_examples.py](training_examples.py)
2. Follow the format: `{'name': str, 'description': str, 'input': ndarray, 'output': ndarray}`
3. Test with `test_integration.py`

## 📄 License

See [LICENSE](LICENSE) file for details.

---

**Built with**: Python 3.12, NumPy, Matplotlib, Websockets

**Architecture**: Modular, object-oriented design with clean separation of concerns

**Status**: Fully functional with 10 preset training examples and comprehensive test suite ✅
