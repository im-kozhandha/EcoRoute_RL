import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(graph, state):
    """Visualize current state of the environment."""
    G = nx.DiGraph()
    for node, neighbors in graph.items():
        for n in neighbors:
            G.add_edge(node, n)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_size=700,
            node_color="skyblue", edge_color="gray")

    # Highlight current node
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[state["current_node"]],
                           node_color="green")

    plt.title(f"Current Node: {state['current_node']}, "
              f"Distance: {state['distance'][0]:.1f}, "
              f"Emissions: {state['emissions'][0]:.2f}")
    plt.show()
