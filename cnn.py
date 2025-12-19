"""Dynamic Neural Network with Interactive Training."""

import asyncio
import json
import time
import websockets
import random
import numpy as np

from neural_network import NeuralNetwork
from interactive_plot import InteractivePlot
import training_examples


# Global network instance
network = None

# Global shutdown flag
shutdown_event = None

# Default mappings (can be overridden by training example selection)
input_mit_mapping = np.array([[0.5, 0], [1, 1], [0, 3], [0, 4]])
output_mit_mapping = np.array([
    [0.321, 5],
    [0.9, 6],
])


def select_training_example():
    """Allow user to select a training example."""
    print("\n" + "="*50)
    print("TRAINING EXAMPLE SELECTION")
    print("="*50)
    print(training_examples.list_examples())
    print("\nOr press Enter to use current custom mapping")
    
    choice = input("\nSelect example (1-10) or press Enter: ").strip()
    
    if choice and choice.isdigit() and 1 <= int(choice) <= 10:
        example = training_examples.get_example(choice)
        print(f"\n✓ Selected: {example['name']}")
        print(f"  {example['description']}")
        return example['input'], example['output']
    else:
        print("\n✓ Using current custom mapping")
        return None, None


async def client_connected_handler(websocket):
    """Client connected."""
    global network

    autorounds = 0
    plot = None
    completion_message_printed = False
    while(True):
        data = [[],[]]
        train_data = {}
        train_data['input'] = np.ndarray(shape=(0,2))
        train_data['output'] = np.ndarray(shape=(0,2))

        if autorounds > 0:
            sel = 'll'
        else:
            print("\n" + "="*60)
            print("   DYNAMIC NEURAL NETWORK - COMMAND MENU")
            print("="*60)
            print("\n📊 TRAINING:")
            print("  ll  - Interactive training (live plot with controls)")
            print("  t   - Single training step")
            print("  s   - Forward pass only (no training)")
            print("\n🔧 NETWORK STRUCTURE:")
            print("  a   - Add a neuron")
            print("  d   - Delete weak synapses")
            print("\n📋 INFORMATION:")
            print("  p   - Print synapse matrix")
            print("  pz  - Print neuron states")
            print("  na  - Print neuron activity")
            print("  ip  - Print input mapping")
            print("  op  - Print output mapping")
            print("\n⚙️  SYSTEM:")
            print("  e   - Exit program")
            print("="*60)
            sel = input("\n→ Enter command: ").strip().lower()

        if(sel == 'e'):
            print("\n👋 Closing connections and exiting...")
            if plot is not None:
                plot.close()
            await websocket.close()
            print("✓ Goodbye!\n")
            # Signal shutdown
            shutdown_event.set()
            break

        elif(sel == 'd'):
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                print('Removing weak synapses...')
                network.prune_weak_synapses(threshold=0.04, max_removals=4)

        elif(sel == 'll'):
            if plot is None:
                # Select training example on first run
                new_input, new_output = select_training_example()
                if new_input is not None:
                    global input_mit_mapping, output_mit_mapping
                    input_mit_mapping = new_input
                    output_mit_mapping = new_output
                    # Keep network mappings in sync with the latest example
                    if network is not None:
                        network.set_mappings(input_mit_mapping, output_mit_mapping)
                
                # Initialize network if needed
                if network is None:
                    network = NeuralNetwork()
                    network.set_mappings(input_mit_mapping, output_mit_mapping)
                
                # Create interactive plot
                plot = InteractivePlot(output_mit_mapping, initial_iterations=100)
                
                # Do initial update to show plot immediately
                plot.update_plot(network.zustand_t1, network.durchgang, 0.0, network.netsize)
                
                autorounds = 1
                completion_message_printed = False
            
            # Check if plot window was closed
            import matplotlib.pyplot as plt
            if not plt.fignum_exists(plot.fig.number):
                print("\n✓ Plot window closed. Returning to menu.\n")
                autorounds = 0
                plot = None
                completion_message_printed = False
                continue

            # Check if we should continue training
            if plot.autorounds[0] > 0 and not plot.is_stopped():
                # Apply inputs
                for inp in input_mit_mapping:
                    network.zustand_t[int(inp[1])] = inp[0]

                # Train
                error = network.step()

                # Pruning based on activity and weak synapses
                if (network.durchgang % 20) == 0:
                    network.prune_inactive_neurons(threshold=0.04, max_removals=5)

                if (network.durchgang % 100) == 0:
                    network.prune_weak_synapses(threshold=0.01, max_removals=4)

                # Conditional growth: only if error above GUI-defined threshold
                if network.durchgang < 3000 and error > plot.grow_error_threshold[0]:
                    if (network.durchgang % 60) == 0:
                        network.add_neuron()
                    if (network.durchgang % 70) == 0:
                        network.grow_network()

                # Periodically try deleting a random weak neuron.
                # The interval (in iterations) is controlled via the GUI slider.
                delete_interval = max(1, int(plot.delete_interval[0]))
                if network.durchgang > 0 and (network.durchgang % delete_interval) == 0:
                    base_error = error
                    saved_state = network.save_state()

                    deleted = network.delete_random_weak_neuron(
                        activity_threshold=0.04,
                        max_synapses=4,
                    )

                    if deleted:
                        # Run a few additional training steps to evaluate impact
                        trial_steps = 5
                        trial_error = base_error
                        for _ in range(trial_steps):
                            trial_error = network.step()

                        # Compare error change against tolerance from GUI (percentage)
                        tol_percent = float(plot.error_tolerance[0])
                        denom = abs(base_error) if abs(base_error) > 1e-8 else 1e-8
                        percent_change = 100.0 * (trial_error - base_error) / denom

                        if percent_change > tol_percent:
                            print(
                                f"Rollback neuron deletion: error increased by "
                                f"{percent_change:5.2f}% > tolerance {tol_percent:5.2f}%"
                            )
                            network.restore_state(saved_state)
                            error = base_error
                        else:
                            print(
                                f"Accepted neuron deletion: error change "
                                f"{percent_change:5.2f}% within tolerance {tol_percent:5.2f}%"
                            )
                            error = trial_error
                
                # Update plot and websocket at the same frequency
                if plot.should_update_plot():
                    plot.update_plot(network.zustand_t1, network.durchgang, error, network.netsize)
                    
                    # Send websocket updates at same frequency as plot updates
                    data[0] = []
                    data[1] = []
                    for i in network.neuro_aktivitaet:
                        data[0].append(float(i))
                    for l in network.synapsenMatrix:
                        data[1].append([int(l[0]), int(l[1]), float(l[2])])
                    
                    train_data['input'] = input_mit_mapping.tolist()
                    train_data['output'] = output_mit_mapping.tolist()
                    
                    await websocket.send(json.dumps(data))
                    await websocket.send(json.dumps(train_data))
                
                # Decrement autorounds
                plot.decrement_rounds()
                
                # Pause handling
                plot.handle_pause()
            
            # Check status
            if plot.is_stopped():
                autorounds = 0
                plot.close()
                plot = None
                completion_message_printed = False
            elif plot.is_complete() and not plot.is_stopped():
                # Print completion message only once
                if not completion_message_printed:
                    plot.print_completion_message()
                    completion_message_printed = True
                
                # Keep looping and checking if "Run More" was clicked
                # Don't go back to menu, stay in this mode
                autorounds = 1  # Keep ll mode active
                # Wait for matplotlib to process button clicks
                import matplotlib.pyplot as plt
                plt.pause(0.1)
                # The loop will continue, and if plot.autorounds[0] > 0, training will resume
                # Reset flag when training resumes
                if plot.autorounds[0] > 0:
                    completion_message_printed = False
                continue  # Skip to next iteration immediately

        elif(sel == 't'):
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                for inp in input_mit_mapping:
                    network.zustand_t[int(inp[1])] = inp[0]
                network.step()
                network.print_output()

        elif(sel == 'a'):
            print('adding neuron \n')
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                network.add_neuron()

        elif(sel == 'p'):
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                network.print_synapses()

        elif(sel == 'pz'):
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                network.print_state()

        elif(sel == 'op'):
            print("output: \n {} \n".format(output_mit_mapping))

        elif(sel == 'ip'):
            print("input: \n {} \n".format(input_mit_mapping))

        elif(sel == 'na'):
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                network.print_activity()

        elif(sel == 's'):
            if network is None:
                print("Error: Network not initialized. Run 'll' first.")
            else:
                zustand_t1_local = np.zeros((len(network.zustand_t), 1))
                for inp in input_mit_mapping:
                    network.zustand_t[int(inp[1])] = inp[0]
                for row in network.synapsenMatrix:
                    if row[1] not in input_mit_mapping[:, 1]:
                        zustand_t1_local[int(row[1])] += network.zustand_t[int(row[0])] * row[2]

                for ind, outOfSum in np.ndenumerate(zustand_t1_local):
                    zustand_t1_local[ind] = 1 / (1 + np.exp(- outOfSum))

                print('only output \n index \n {} \n loesung: \n {} \n'.format(
                    output_mit_mapping[:, 1],
                    zustand_t1_local[output_mit_mapping[:, 1].astype(int)]
                ))

        else:
            if sel:
                print(f"\n⚠️  Unknown command: '{sel}'")
                print("Type a valid command from the menu above.\n")

        
        train_data['input'] = input_mit_mapping
        train_data['output'] = output_mit_mapping
                

        train_data['input'] = train_data['input'].tolist()
        train_data['output'] = train_data['output'].tolist()
        
        neuro_aktivitaet_to_send = network.neuro_aktivitaet if network else []
        for i in neuro_aktivitaet_to_send:
            data[0].append(i)

        synapsenMatrix_to_send = network.synapsenMatrix if network else []
        for l in synapsenMatrix_to_send:
            data[1].append([l[0],l[1],l[2]])

        await websocket.send(json.dumps(data))
        await websocket.send(json.dumps(train_data))


        # for row in synapsenMatrix:
        #     for inp in input_mit_mapping:
        #         zustand_t[inp[1]]=inp[0]
        #     zustand_t1[row[1]] = 1/(1+np.exp(-(zustand_t[row[1]] * row[2])))

        
        # await websocket.send(json.dumps(train_data))
    
# synapsenMatrix = np.array([
#     [0, 2, 0.2],
#     [0, 1, 0.1],
#     [0, 4, 0.1],
#     [0, 5, 0.1],
#     [0, 6, 0.1],
#     [1, 4, 0.5],
#     [1, 5, 0.5],
#     [1, 6, 0.5],
#     [1, 3, 0.5],
#     [1, 7, 0.5],
#     [1, 8, 0.5],
#     [1, 9, 0.5],
#     [2, 7, 0.5],
#     [2, 8, 0.5],
#     [2, 9, 0.5],
#     [3, 7, 0.5],
#     [3, 8, 0.5],
#     [3, 9, 0.5],
#     [4, 7, 0.5],
#     [4, 8, 0.5],
#     [4, 9, 0.5],
#     [5, 7, 0.5],
#     [5, 8, 0.5],
#     [5, 9, 0.5],
#     [6, 7, 0.5],
#     [6, 8, 0.5],
#     [6, 9, 0.5],
#     [7, 8, 0.5],
#     [7, 9, 0.5],
#     [2, 1, 0.5]
# ])

# input are nodes 0-4 of the network, output are nodes 7-12 of the nw
# first entry is the value, second entry is the noden


# print("synapsenMatrix: {}".format(synapsenMatrix))






async def main():
    global shutdown_event
    shutdown_event = asyncio.Event()
    
    async with websockets.serve(client_connected_handler, "", 5678):
        print("WebSocket server started on port 5678")
        print("Open visual.html in your browser to see the network visualization\n")
        await shutdown_event.wait()



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!\n")