import networkx as nx
import random
import os

# ==================== Your Original Generators ====================

def generate_debug_graph():
    """Returns a small, structured graph useful for step-by-step debugging."""
    G = nx.DiGraph()
    edges = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 1),
        (0, 3, 10),
        (3, 4, 2)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G

def generate_mid_sized_graph(n=1000, m=3000, directed=False):
    """Returns a synthetic mid-sized graph with weights."""
    G = nx.gnm_random_graph(n=n, m=m, directed=directed)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 20)
    return G

def generate_directed_weighted_graph(n=500, m=2000):
    """Returns a directed graph with non-negative weights for Dijkstra testing."""
    G = nx.gnm_random_graph(n=n, m=m, directed=True)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 50)
    return G

def generate_disconnected_graph(n=500, edge_prob=0.005):
    """Returns a graph likely to have disconnected components."""
    G = nx.erdos_renyi_graph(n=n, p=edge_prob)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 15)
    return G

# ==================== Saving + Loading ====================

def save_graph_to_txt(G, filename):
    """Save a graph to file in edge list format with weights."""
    with open(filename, "w") as f:
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1)
            f.write(f"{u} {v} {w}\n")

def load_graph_from_txt(filename, directed=False):
    """Load a graph from a txt file."""
    G = nx.DiGraph() if directed else nx.Graph()
    with open(filename, "r") as f:
        for line in f:
            u, v, w = line.strip().split()
            G.add_edge(int(u), int(v), weight=float(w))
    return G

# ==================== Batch Generator ====================

def generate_graph_batch(
    output_dir="graph_datasets",
    num_graphs=50,
    min_nodes=50,
    max_nodes=500,
    directed_ratio=0.5,
    disconnected_chance=0.3
):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        is_directed = random.random() < directed_ratio
        is_disconnected = random.random() < disconnected_chance

        if is_disconnected:
            p = round(random.uniform(0.001, 0.01), 4)
            G = nx.erdos_renyi_graph(n=n, p=p, directed=is_directed)
            graph_type = "disconnected"
        else:
            max_possible_edges = n * (n - 1) if is_directed else n * (n - 1) // 2
            m = random.randint(n, max_possible_edges // 10)
            G = nx.gnm_random_graph(n=n, m=m, directed=is_directed)
            graph_type = "directed" if is_directed else "undirected"

        # Add weights
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.randint(1, 20)

        # Create subfolder
        subfolder = os.path.join(output_dir, graph_type)
        os.makedirs(subfolder, exist_ok=True)

        # Save graph
        filename = f"{graph_type}_graph_{i}_n{n}_e{G.number_of_edges()}.txt"
        filepath = os.path.join(subfolder, filename)
        save_graph_to_txt(G, filepath)

        print(f"[+] Saved {filepath}")

# ==================== Runner ====================

if __name__ == "__main__":
    # Save the debug graph in its own subfolder
    debug_dir = "graph_datasets/debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_graph = generate_debug_graph()
    save_graph_to_txt(debug_graph, os.path.join(debug_dir, "debug_graph.txt"))

    # Generate 50 varied graphs, each saved in its appropriate subfolder
    generate_graph_batch(
        output_dir="graph_datasets",
        num_graphs=50,
        min_nodes=100,
        max_nodes=1000,
        directed_ratio=0.5,
        disconnected_chance=0.25
    )
