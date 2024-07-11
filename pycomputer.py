import json
import os
from typing import Dict
# import ast
import time
from formatters import nice_json_dump


class Computer:
    """A class to represent a computer object that can load graphs, 
    data, compute, and store graphs to files.

    Attributes:
        current_graph: A dictionary representing the current graph.
        current_data: A list representing the current data.

    Methods:
        loadgraph: Loads a graph into the computer object.
        loaddata: Loads training data into the computer object.
        compute: Computes the forward and backward pass of the neural network.
        store_current2file: Stores the current graph to a file in the current directory.
        load_fromfile: Loads a graph from a file in the current directory.

    """

    def __init__(self, graph: dict = None) -> None:
        """Initializes the computer object with a graph.        

        @param graph: The graph to initialize the computer with.
        @return: None
        """
        if graph is None:
            graph = {"nodes": [], "edges": []}
        self.current_graph = graph
        self.current_data = []

    def loadgraph(self, graph: dict) -> Dict[str, int]:
        """Loads a graph into the computer object.

        @param graph: The graph to load into the computer object.

        @return: Dictionary containing the number of nodes and edges in the graph.
        """
        self.current_graph = graph
        print("Loading graph", nice_json_dump(graph))
        
        return {"#nodes": len(graph["nodes"]), "#edges": len(graph["edges"])}

    def loaddata(self, data: list) -> int:
        """Loads training data into the computer object.

        @param data: The data to load into the computer object.
        @return: The number of data points loaded.
        """
        self.current_data = data
        print("Loading data", data)

    def compute(self, iterations=1) -> int:
        """Computes the forward and backward pass of the neural network.

        @param iterations: The number of iterations to compute.
        @return: The average time taken to compute the forward and backward pass.
        """

        print("Computing", iterations)
        start = time.perf_counter_ns()

        # #TODO: Implement the forward and backward pass of the neural network here.

        # def ReLU():
        #     pass

        # def differentiate_ReLU():
        #     pass

        # def add_synapse():
        #     pass

        # def compute_synapse():
        #     pass

        # def forward_pass():
        #     pass

        # def backward_pass():
        #     pass

        for _ in range(iterations):
            time.sleep(0.1)
        end = time.perf_counter_ns()
        return (end - start)/float(iterations)

    def store_current2file(self, file: str):
        """Stores the current graph to a file in the current directory.
        The file is stored in JSON format.

        @param file: The name of the file to store the graph in.                
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(current_dir, file)
        with open(file, "w", encoding="utf-8") as f:
            f.write(nice_json_dump(self.current_graph))
            print("stored file", file)

    def load_fromfile(self, file: str):
        """Loads a graph from a file in the current directory.
        The file should be in JSON format.

        @param file: The name of the file to load the graph from.        
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(current_dir, file)
        with open(file, "r", encoding="utf-8") as f:
            self.current_graph = json.load(f)
        #     self.current_graph = ast.literal_eval(f.read())
        #     print("loaded file", file)
