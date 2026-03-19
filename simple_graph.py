import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button

# =========================
# CONFIG
# =========================
UPDATE_FREQUENCY = 0.02
LEARNING_RATE = 0.01
MAX_HISTORY = 300

USE_INPUT_OVERRIDE = True
USE_TARGET_OVERWRITE = True
FREE_RUN = False  # disables overwrite but keeps training

NUM_NODES = 4
INPUT_NODE = 0
OUTPUT_NODE = 3

# Fully connected graph (no self loops)
rng = np.random.default_rng(42)
EDGES = [(i, j, float(rng.uniform(0, 1)))
         for i in range(NUM_NODES)
         for j in range(NUM_NODES) if i != j]

# =========================
# ACTIVATION
# =========================
def relu(x): return max(0.0, x)
def relu_deriv(x): return 1.0 if x > 0 else 0.0

# =========================
# FUNCTION THREAD (x → sin(x))
# =========================
class FunctionThread:
    def __init__(self):
        self.x = 0.0
        self.sinx = 0.0
        self.running = True

    def run(self):
        while self.running:
            self.sinx = np.sin(self.x)
            self.x += 0.05
            time.sleep(UPDATE_FREQUENCY)

func_thread = FunctionThread()
thread = threading.Thread(target=func_thread.run, daemon=True)
thread.start()

# =========================
# GRAPH STRUCTURE
# =========================
class Edge:
    def __init__(self, src, tgt, w):
        self.src, self.tgt = src, tgt
        self.w = w
        self.grad = 0.0

class Node:
    def __init__(self, idx):
        self.idx = idx
        self.in_edges = []
        self.out_edges = []
        self.value = 0.0
        self.pre_activation = 0.0
        self.delta = 0.0

nodes = [Node(i) for i in range(NUM_NODES)]
edges = []

for src, tgt, w in EDGES:
    e = Edge(src, tgt, w)
    edges.append(e)
    nodes[src].out_edges.append(e)
    nodes[tgt].in_edges.append(e)

# =========================
# VISUALIZATION
# =========================
G = nx.DiGraph()
for i in range(NUM_NODES):
    G.add_node(i)
for e in edges:
    G.add_edge(e.src, e.tgt, weight=e.w)

pos = nx.spring_layout(G, seed=1)

plt.ion()
fig, (ax_graph, ax_func) = plt.subplots(2, 1, figsize=(10, 8))
fig2, axs = plt.subplots(2, 2, figsize=(8, 6))
axs = axs.flatten()

history = [np.zeros(MAX_HISTORY) for _ in range(NUM_NODES)]
target_history = np.zeros(MAX_HISTORY)
loss_history = np.zeros(MAX_HISTORY)
idx = 0

node_lines = [axs[i].plot(history[i])[0] for i in range(NUM_NODES)]
target_line, = ax_func.plot(target_history, label='sin(x)')
loss_line, = ax_func.plot(loss_history, label='loss')
ax_func.legend()

# =========================
# BUTTONS
# =========================
ax_b1 = plt.axes([0.65, 0.01, 0.1, 0.05])
ax_b2 = plt.axes([0.76, 0.01, 0.1, 0.05])
ax_b3 = plt.axes([0.87, 0.01, 0.1, 0.05])

btn_input = Button(ax_b1, 'Input ON')
btn_target = Button(ax_b2, 'Target ON')
btn_free = Button(ax_b3, 'Free OFF')

def toggle_input(event):
    global USE_INPUT_OVERRIDE
    USE_INPUT_OVERRIDE = not USE_INPUT_OVERRIDE
    btn_input.label.set_text(f"Input {'ON' if USE_INPUT_OVERRIDE else 'OFF'}")

def toggle_target(event):
    global USE_TARGET_OVERWRITE
    USE_TARGET_OVERWRITE = not USE_TARGET_OVERWRITE
    btn_target.label.set_text(f"Target {'ON' if USE_TARGET_OVERWRITE else 'OFF'}")

def toggle_free(event):
    global FREE_RUN
    FREE_RUN = not FREE_RUN
    btn_free.label.set_text(f"Free {'ON' if FREE_RUN else 'OFF'}")

btn_input.on_clicked(toggle_input)
btn_target.on_clicked(toggle_target)
btn_free.on_clicked(toggle_free)

# =========================
# UPDATE
# =========================
def update():
    # Forward
    for node in nodes:
        total = sum(nodes[e.src].value * e.w for e in node.in_edges)
        node.pre_activation = total
        node.value = total if node.idx == OUTPUT_NODE else relu(total)

    if USE_INPUT_OVERRIDE:
        nodes[INPUT_NODE].value = func_thread.x

    target = func_thread.sinx
    output = nodes[OUTPUT_NODE].value

    loss = 0.5 * (output - target) ** 2

    # Output delta (linear output)
    nodes[OUTPUT_NODE].delta = (output - target)

    # Backprop
    for node in reversed(nodes[:-1]):
        total = sum(e.w * nodes[e.tgt].delta for e in node.out_edges)
        node.delta = total * relu_deriv(node.pre_activation)

    # Gradients + update
    for e in edges:
        e.grad = nodes[e.src].value * nodes[e.tgt].delta
        e.w -= LEARNING_RATE * e.grad

    # Overwrite ONLY for visualization (unless free run)
    if USE_TARGET_OVERWRITE and not FREE_RUN:
        nodes[OUTPUT_NODE].value = target

    return loss

# =========================
# DRAW
# =========================
def autoscale(ax, data):
    dmin, dmax = np.min(data), np.max(data)
    if dmin == dmax:
        dmax += 1e-3
    ax.set_ylim(dmin, dmax)

def draw(loss):
    global idx
    idx = (idx + 1) % MAX_HISTORY

    target_history[idx] = func_thread.sinx
    loss_history[idx] = loss

    for i in range(NUM_NODES):
        history[i][idx] = nodes[i].value
        node_lines[i].set_ydata(history[i])
        autoscale(axs[i], history[i])
        axs[i].set_title(f'Node {i}')

    target_line.set_ydata(target_history)
    loss_line.set_ydata(loss_history)
    autoscale(ax_func, np.concatenate([target_history, loss_history]))

    if idx % 5 == 0:
        ax_graph.clear()
        values = [nodes[i].value for i in G.nodes]
        weights = [e.w for e in edges]

        nx.draw(G, pos, ax=ax_graph,
                with_labels=True,
                node_color=values,
                edge_color=weights,
                cmap=plt.cm.viridis,
                width=[abs(w) * 2 for w in weights])

    fig.canvas.draw_idle()
    fig2.canvas.draw_idle()
    plt.pause(0.001)

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        loss = update()
        draw(loss)
        time.sleep(UPDATE_FREQUENCY)
except KeyboardInterrupt:
    func_thread.running = False
    print("Stopped.")