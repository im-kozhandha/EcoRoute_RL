import numpy as np

def build_graph(num_nodes=4):
    """Return adjacency matrix and distances between nodes."""
    # Example fully connected graph with dummy distances
    distances = np.full((num_nodes, num_nodes), np.inf)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i][j] = np.random.randint(5, 15)  # random distance
    graph = {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}
    return graph, distances

def calculate_emissions(distance, emission_rate=0.2):
    """Simple linear emissions model."""
    return distance * emission_rate
