"""Training examples for the neural network."""

import numpy as np

# Dictionary of training examples
TRAINING_EXAMPLES = {
    '1': {
        'name': 'Simple Binary (2 outputs)',
        'description': 'Learn two simple output values',
        'input': np.array([[0.5, 0], [1, 1], [0, 3], [0, 4]]),
        'output': np.array([[0.321, 5], [0.9, 6]])
    },
    '2': {
        'name': 'XOR Gate',
        'description': 'Classic XOR problem',
        'input': np.array([[0, 0], [1, 1], [0, 2], [1, 3]]),
        'output': np.array([[0, 4], [1, 5]])
    },
    '3': {
        'name': 'AND Gate',
        'description': 'Logical AND operation',
        'input': np.array([[0, 0], [1, 1], [0, 2], [1, 3]]),
        'output': np.array([[0, 4]])
    },
    '4': {
        'name': 'Multi-Target (6 outputs)',
        'description': 'Learn multiple output patterns',
        'input': np.array([[0.5, 0], [1, 1], [0, 3], [0, 4]]),
        'output': np.array([
            [0.7, 7],
            [0, 8],
            [0.5, 9],
            [0.1, 10],
            [0.134, 11],
            [0.321, 12]
        ])
    },
    '5': {
        'name': 'Linear Pattern',
        'description': 'Linear relationship between inputs and outputs',
        'input': np.array([[0.2, 0], [0.4, 1], [0.6, 2], [0.8, 3]]),
        'output': np.array([[0.25, 5], [0.5, 6], [0.75, 7]])
    },
    '6': {
        'name': 'Symmetric Pattern',
        'description': 'Symmetric output values',
        'input': np.array([[0.5, 0], [0.5, 1], [0.5, 2], [0.5, 3]]),
        'output': np.array([[0.3, 5], [0.5, 6], [0.7, 7], [0.5, 8], [0.3, 9]])
    },
    '7': {
        'name': 'Binary Counter',
        'description': 'Binary counting pattern',
        'input': np.array([[1, 0], [1, 1], [1, 2], [1, 3]]),
        'output': np.array([[0, 5], [0, 6], [1, 7], [1, 8]])
    },
    '8': {
        'name': 'High Precision',
        'description': 'Multiple precise decimal targets',
        'input': np.array([[0.123, 0], [0.456, 1], [0.789, 2]]),
        'output': np.array([
            [0.111, 5],
            [0.222, 6],
            [0.333, 7],
            [0.444, 8],
            [0.555, 9],
            [0.666, 10],
            [0.777, 11],
            [0.888, 12],
            [0.999, 13]
        ])
    },
    '9': {
        'name': 'Sparse Pattern',
        'description': 'Few active inputs, many outputs',
        'input': np.array([[1, 0], [0.5, 1]]),
        'output': np.array([[0.1, 5], [0.3, 6], [0.5, 7], [0.7, 8], [0.9, 9]])
    },
    '10': {
        'name': 'Random Challenge',
        'description': 'Random target values for exploration',
        'input': np.array([[0.5, 0], [1, 1], [0, 3], [0, 4]]),
        'output': np.array([
            [np.random.random(), 5],
            [np.random.random(), 6],
            [np.random.random(), 7],
            [np.random.random(), 8]
        ])
    }
}


def get_example(example_id):
    """Get a training example by ID."""
    if example_id in TRAINING_EXAMPLES:
        return TRAINING_EXAMPLES[example_id]
    return None


def list_examples():
    """Return a formatted string listing all examples."""
    lines = ["\n=== Available Training Examples ===\n"]
    for key in sorted(TRAINING_EXAMPLES.keys(), key=int):
        example = TRAINING_EXAMPLES[key]
        lines.append(f"{key}. {example['name']}")
        lines.append(f"   {example['description']}")
        lines.append(f"   Inputs: {len(example['input'])}, Outputs: {len(example['output'])}\n")
    return '\n'.join(lines)


def get_example_prompt():
    """Return a prompt string for selecting an example."""
    return (
        "\nSelect training example:\n"
        "1-10 for preset examples\n"
        "'list' to show all examples\n"
        "'custom' to use current mapping\n"
        "Choice: "
    )
