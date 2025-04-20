
from collections import defaultdict
import heapq
import math
import time
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


#creating the graph
class Graph:
    def __init__(self, directed=True):
    # we will store the graph in form of disctionary
        self.graph = defaultdict(list)
        # we will also create its reverse for backward edges which will be used in bidirectional dijkstra
        self.reverse_graph = defaultdict(list) if directed else self.graph
        # we will be working on directed graphs
        self.directed = directed
        #and we have to keep track of the vertices we have visisted
        self.vertices = set()
#function to add the edges to the graph
    def add_edge(self, u, v, weight):
        # we will be working with directed graph and in case of undirected we will add edges on both cases
        self.graph[u].append((v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
        if self.directed:
            self.reverse_graph[v].append((u, weight))
        else:
            self.graph[v].append((u, weight))
    #get the neighbour of vertice

    def get_neighbors(self, u, reverse=False):
        return self.reverse_graph[u] if reverse else self.graph[u]
#code for bidirectional 
def bidirectional_dijkstra(graph, source, target):
    # if the source or target are not in the graph return 0
    if source not in graph.vertices or target not in graph.vertices:
        return [], math.inf, 0
    #if source is queal to target then we will simply return 0
    if source == target:
        return [source], 0, 0
    # mu is the smallest possible distance between the two path
    mu = math.inf
    # the edge where the forward and backward search meets
    e_mid = None
    #us is the start of the source
    #ut is the start of the destination
    us, ut = source, target
    #keeping tarck of edges we have visiyed
    edge_count = 0
    #initualising the forward tarcking 
    d_forward = {v: math.inf for v in graph.vertices}
    d_forward[source] = 0
    pred_forward = {source: None}
    open_forward = [(0, source)]
    closed_forward = set()
    #initialisation for the backward edge
    d_backward = {v: math.inf for v in graph.vertices}
    d_backward[target] = 0
    pred_backward = {target: None}
    open_backward = [(0, target)]
    closed_backward = set()

    def relax_edge(u, v, weight, direction):
        #mu -> shortest distance we have found so far
        # e_mid -> the edge where the forward and backward edge meeets 

        nonlocal mu, e_mid, edge_count
         
        edge_count += 1
        # now we check the distance of the edge based on whether it is forward search or backward 
        d = d_forward if direction == 'forward' else d_backward
        pred = pred_forward if direction == 'forward' else pred_backward
        open_heap = open_forward if direction == 'forward' else open_backward
        closed_opposite = closed_backward if direction == 'forward' else closed_forward

        new_dist = d[u] + weight
        if new_dist < d[v]:
            d[v] = new_dist
            pred[v] = u
            heapq.heappush(open_heap, (new_dist, v))
            if v in closed_opposite:
                path_len = d_forward[v] + d_backward[v]
                if path_len < mu:
                    mu = path_len
                    e_mid = (u, v, weight) if direction == 'forward' else (v, u, weight)
    #start seaching from the start
    direction = 'forward'
    # and we continiye doing it 
    while open_forward and open_backward:
        #until when the sum of the forward and backward is the least shortest distance we have found
        if d_forward[us] + d_backward[ut] >= mu and mu != math.inf:
            break
        #if we are iterating in forward direction and we check if the queue is empty
        if direction == 'forward':
            if not open_forward:
                break
            #we pop from the priority queue
            dist, u = heapq.heappop(open_forward)
            # if the node is in the visited vertice then we continue
            if u in closed_forward:
                continue
            #else we add it to the visited queue
            closed_forward.add(u)
            us = u
            #then start iterating over the neighbours of the u and see if they are in the visited vertices if not then we add then to relax edge in forward directions
            for v, weight in graph.get_neighbors(u):
                if v not in closed_forward:
                    relax_edge(u, v, weight, 'forward')
            #now search backward
            direction = 'backward'
        else:
            if not open_backward:
                break
            dist, u = heapq.heappop(open_backward)
            if u in closed_backward:
                continue
            closed_backward.add(u)
            ut = u
            for v, weight in graph.get_neighbors(u, reverse=graph.directed):
                if v not in closed_backward:
                    relax_edge(u, v, weight, 'backward')
            direction = 'forward'

    if mu == math.inf or e_mid is None:
        return [], math.inf, edge_count

    u, v, _ = e_mid
    path = []
    curr = u
    while curr is not None:
        path.append(curr)
        curr = pred_forward[curr]
    path = path[::-1]
    path.append(v)
    curr = pred_backward[v]
    while curr is not None:
        path.append(curr)
        curr = pred_backward[curr]

    return path, mu, edge_count


#unidirectional dijsktra 
def dijkstra(graph, source, target):
    # if either source or target not in the graph return none
    if source not in graph.vertices or target not in graph.vertices:
        return [], math.inf, 0
        # if source is the target then the distance will be 0
    if source == target:
        return [source], 0, 0
    # making edge counter
    edge_count = 0
    #initualising the distance except for the source as infinity
    d = {v: math.inf for v in graph.vertices}
    d[source] = 0
    # keeping the predecessor of the source as none
    pred = {source: None}
    #adding the source into the priority queue
    pq = [(0, source)]
    # making a visisted to keep track of all of the visited nodes
    visited = set()
    # this will continue happening until our priority queue is empty
    while pq:

        dist, u = heapq.heappop(pq)
        # if the node is in visited we will skip that
        if u in visited:
            continue
        # else add it to the visited list
        visited.add(u)
        # if the node is the target we will stop searching
        if u == target:
            break
        #else we will explor the neighbours of the nodes and the edge count will increment even for the nodes whcih are duplicated or are already visited
        for v, weight in graph.get_neighbors(u):
            #increase the edge count
            edge_count += 1
            # if the node is not visited we check the diatcne from its parents if its is smaller then we will update it
            if v not in visited and d[u] + weight < d[v]:
                d[v] = d[u] + weight
                #update the predecessor
                pred[v] = u
                # add the neighbour in the priority queue
                heapq.heappush(pq, (d[v], v))
    # if the target's distance is still infinity which means there is no way to go to destination then we will just return empty list
    if d[target] == math.inf:
        return [], math.inf, edge_count
    # reversing the path to get the original path
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        curr = pred[curr]
    return path[::-1], d[target], edge_count

def networkx_dijkstra(graph, source, target):
    G = nx.DiGraph() if graph.directed else nx.Graph()
    for u in graph.vertices:
        for v, w in graph.get_neighbors(u):
            G.add_edge(u, v, weight=w)
    
    try:
        length, path = nx.single_source_dijkstra(G, source, target, weight='weight')
        return list(path), length, 0
    except nx.NetworkXNoPath:
        return [], math.inf, 0

def generate_random_graph(n, m, directed=True, seed=42):
    random.seed(seed)
    g = Graph(directed=directed)
    for i in range(n-1):
        g.add_edge(i, i+1, random.randint(1, 100))
    for _ in range(m - (n-1)):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            weight = random.randint(1, 100)
            g.add_edge(u, v, weight)
    return g

def run_performance_test(graph, source, target, iterations=100, name="Graph"):
    print(f"\n{name} (from {source} to {target}):")
    
    nx_path, nx_length, _ = networkx_dijkstra(graph, source, target)
    print(f"NetworkX: Length={nx_length}")

    #calclating the time for the bidirectional
    total_bidi_time = 0
    total_bidi_edges = 0
    bidi_path = None
    bidi_length = None
    for i in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = bidirectional_dijkstra(graph, source, target)
        total_bidi_time += time.perf_counter() - start_time
        total_bidi_edges += edges
        if bidi_path is None:
            bidi_path, bidi_length = path, length
    #the timing is in ms
    avg_bidi_time = total_bidi_time / iterations * 1000 
    avg_bidi_edges = total_bidi_edges / iterations
    print(f"Bidirectional Dijkstra: Length={bidi_length}")
    print(f"  Avg Time: {avg_bidi_time:.2f} ms, Avg Edges: {avg_bidi_edges:.1f}")

    #for unidirectional 
    total_dijk_time = 0
    total_dijk_edges = 0
    dijk_path = None
    dijk_length = None
    for _ in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = dijkstra(graph, source, target)
        total_dijk_time += time.perf_counter() - start_time
        total_dijk_edges += edges
        if dijk_path is None:
            dijk_path, dijk_length = path, length

    avg_dijk_time = total_dijk_time / iterations * 1000  # ms
    avg_dijk_edges = total_dijk_edges / iterations
    print(f"Unidirectional Dijkstra: Length={dijk_length}")
    print(f"  Avg Time: {avg_dijk_time:.2f} ms, Avg Edges: {avg_dijk_edges:.1f}")

def test_algorithms():
    #Small graph
    g_small = Graph(directed=True)
    g_small.add_edge(1, 2, 4)
    g_small.add_edge(1, 3, 2)
    g_small.add_edge(2, 4, 3)
    g_small.add_edge(3, 2, 1)
    g_small.add_edge(3, 4, 5)
    g_small.add_edge(4, 5, 2)
    g_small.add_edge(2, 5, 10)
    print("Small Directed Graph (5 vertices, 7 edges):")
    run_performance_test(g_small, 1, 5, iterations=100, name="Small Directed")

    #Medium graph
    g_medium = generate_random_graph(n=10000, m=50000, directed=True)
    print("\nMedium Directed Graph (10,000 vertices, 50,000 edges):")
    run_performance_test(g_medium, 0, 9999, iterations=10, name="Medium Directed")

    # Large graph
    g_large = generate_random_graph(n=50000, m=250000, directed=True)
    print("\nLarge Directed Graph (50,000 vertices, 250,000 edges):")
    run_performance_test(g_large, 0, 49999, iterations=10, name="Large Directed")

    # Very large graph
    g_very_large = generate_random_graph(n=100000, m=500000, directed=True)
    print("\nVery Large Directed Graph (100,000 vertices, 500,000 edges):")
    run_performance_test(g_very_large, 0, 99999, iterations=10, name="Very Large Directed")



def plot_comparison_trend(results):
    # Extract graph names and edge counts
    graph_names = [res['name'] for res in results]
    uni_edges = [res['uni_edges'] for res in results]
    bidi_edges = [res['bidi_edges'] for res in results]
    
    # Set up trend line graph
    x = np.arange(len(graph_names))  # Positions for points
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, uni_edges, marker='o', linestyle='-', color='skyblue', label='Unidirectional')
    plt.plot(x, bidi_edges, marker='s', linestyle='--', color='lightcoral', label='Bidirectional')
    
    # Customize graph
    plt.xlabel('Graph Type')
    plt.ylabel('Average Edges Checked')
    plt.title('Trend of Edges Checked: Unidirectional vs. Bidirectional Dijkstra’s')
    plt.xticks(x, graph_names, rotation=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels near points
    for i, v in enumerate(uni_edges):
        plt.text(i, v + 0.57, f'{v:.1f}', ha='center', va='bottom')
    for i, v in enumerate(bidi_edges):
        plt.text(i, v - 0.5, f'{v:.1f}', ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig('comparison_edges_trend.png')
    plt.close()

# Modified run_performance_test to collect results (integrates with your code)
def run_performance_test(graph, source, target, iterations=100, name="Graph"):
    print(f"\n{name} (from {source} to {target}):")
    
    nx_path, nx_length, _ = networkx_dijkstra(graph, source, target)
    print(f"NetworkX: Length={nx_length}")

    total_bidi_time = 0
    total_bidi_edges = 0
    bidi_path = None
    bidi_length = None
    for i in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = bidirectional_dijkstra(graph, source, target)
        total_bidi_time += time.perf_counter() - start_time
        total_bidi_edges += edges
        if bidi_path is None:
            bidi_path, bidi_length = path, length
    avg_bidi_time = total_bidi_time / iterations * 1000
    avg_bidi_edges = total_bidi_edges / iterations
    print(f"Bidirectional Dijkstra: Length={bidi_length}")
    print(f"  Avg Time: {avg_bidi_time:.2f} ms, Avg Edges: {avg_bidi_edges:.1f}")

    total_dijk_time = 0
    total_dijk_edges = 0
    dijk_path = None
    dijk_length = None
    for _ in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = dijkstra(graph, source, target)
        total_dijk_time += time.perf_counter() - start_time
        total_dijk_edges += edges
        if dijk_path is None:
            dijk_path, dijk_length = path, length
    avg_dijk_time = total_dijk_time / iterations * 1000
    avg_dijk_edges = total_dijk_edges / iterations
    print(f"Unidirectional Dijkstra: Length={dijk_length}")
    print(f"  Avg Time: {avg_dijk_time:.2f} ms, Avg Edges: {avg_dijk_edges:.1f}")
    
    # Return results for plotting
    return {
        'name': name,
        'uni_edges': avg_dijk_edges,
        'bidi_edges': avg_bidi_edges
    }

def test_algorithms_with_plot():
    results = []
    
    # Small graph
    g_small = Graph(directed=True)
    g_small.add_edge(1, 2, 4)
    g_small.add_edge(1, 3, 2)
    g_small.add_edge(2, 4, 3)
    g_small.add_edge(3, 2, 1)
    g_small.add_edge(3, 4, 5)
    g_small.add_edge(4, 5, 2)
    g_small.add_edge(2, 5, 10)
    results.append(run_performance_test(g_small, 1, 5, iterations=100, name="Small Directed"))

    # Medium graph
    g_medium = generate_random_graph(n=10000, m=50000, directed=True)
    results.append(run_performance_test(g_medium, 0, 9999, iterations=10, name="Medium Directed"))

    # Large graph
    g_large = generate_random_graph(n=50000, m=250000, directed=True)
    results.append(run_performance_test(g_large, 0, 49999, iterations=10, name="Large Directed"))

    # Very large graph
    g_very_large = generate_random_graph(n=100000, m=500000, directed=True)
    results.append(run_performance_test(g_very_large, 0, 99999, iterations=10, name="Very Large Directed"))

    # Plot the trend line graph
    plot_comparison_trend(results)




def plot_comparison_time_trend(results):
    # Extract graph names and time values
    graph_names = [res['name'] for res in results]
    uni_times = [res['uni_time'] for res in results]
    bidi_times = [res['bidi_time'] for res in results]
    
    # Set up trend line graph
    x = np.arange(len(graph_names))  # Positions for points
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, uni_times, marker='o', linestyle='-', color='skyblue', label='Unidirectional')
    plt.plot(x, bidi_times, marker='s', linestyle='--', color='lightcoral', label='Bidirectional')
    
    # Customize graph
    plt.xlabel('Graph Type')
    plt.ylabel('Average Execution Time (ms)')
    plt.title('Trend of Execution Times: Unidirectional vs. Bidirectional Dijkstra’s')
    plt.xticks(x, graph_names, rotation=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels near points
    for i, v in enumerate(uni_times):
        plt.text(i, v + 0.05 * v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(bidi_times):
        plt.text(i, v - 0.05 * v, f'{v:.2f}', ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig('comparison_times_trend_time.png')
    plt.close()

# Modified run_performance_test to collect results (integrates with your code)
def run_performance_test_time(graph, source, target, iterations=100, name="Graph"):
    print(f"\n{name} (from {source} to {target}):")
    
    nx_path, nx_length, _ = networkx_dijkstra(graph, source, target)
    print(f"NetworkX: Length={nx_length}")

    total_bidi_time = 0
    total_bidi_edges = 0
    bidi_path = None
    bidi_length = None
    for i in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = bidirectional_dijkstra(graph, source, target)
        total_bidi_time += time.perf_counter() - start_time
        total_bidi_edges += edges
        if bidi_path is None:
            bidi_path, bidi_length = path, length
    avg_bidi_time = total_bidi_time / iterations * 1000
    avg_bidi_edges = total_bidi_edges / iterations
    print(f"Bidirectional Dijkstra: Length={bidi_length}")
    print(f"  Avg Time: {avg_bidi_time:.2f} ms, Avg Edges: {avg_bidi_edges:.1f}")

    total_dijk_time = 0
    total_dijk_edges = 0
    dijk_path = None
    dijk_length = None
    for _ in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = dijkstra(graph, source, target)
        total_dijk_time += time.perf_counter() - start_time
        total_dijk_edges += edges
        if dijk_path is None:
            dijk_path, dijk_length = path, length
    avg_dijk_time = total_dijk_time / iterations * 1000
    avg_dijk_edges = total_dijk_edges / iterations
    print(f"Unidirectional Dijkstra: Length={dijk_length}")
    print(f"  Avg Time: {avg_dijk_time:.2f} ms, Avg Edges: {avg_dijk_edges:.1f}")
    
    # Return results for plotting
    return {
        'name': name,
        'uni_time': avg_dijk_time,
        'bidi_time': avg_bidi_time,
        'uni_edges': avg_dijk_edges,
        'bidi_edges': avg_bidi_edges
    }

def test_algorithms_with_plot_time():
    results = []
    
    # Small graph
    g_small = Graph(directed=True)
    g_small.add_edge(1, 2, 4)
    g_small.add_edge(1, 3, 2)
    g_small.add_edge(2, 4, 3)
    g_small.add_edge(3, 2, 1)
    g_small.add_edge(3, 4, 5)
    g_small.add_edge(4, 5, 2)
    g_small.add_edge(2, 5, 10)
    results.append(run_performance_test_time(g_small, 1, 5, iterations=100, name="Small Directed"))

    # Medium graph
    g_medium = generate_random_graph(n=10000, m=50000, directed=True)
    results.append(run_performance_test_time(g_medium, 0, 9999, iterations=10, name="Medium Directed"))

    # Large graph
    g_large = generate_random_graph(n=50000, m=250000, directed=True)
    results.append(run_performance_test_time(g_large, 0, 49999, iterations=10, name="Large Directed"))

    # Very large graph
    g_very_large = generate_random_graph(n=100000, m=500000, directed=True)
    results.append(run_performance_test_time(g_very_large, 0, 99999, iterations=10, name="Very Large Directed"))

    # Plot the trend line graph
    plot_comparison_time_trend(results)

# if __name__ == "__main__":
#     test_algorithms_with_plot()
#     test_algorithms_with_plot_time()


def astar(graph, source, target):
    # Handle invalid inputs
    if source not in graph.vertices or target not in graph.vertices:
        return [], math.inf, 0
    if source == target:
        return [source], 0, 0
    
    # Initialize data structures
    edge_count = 0
    g = {v: math.inf for v in graph.vertices}  # Cost from source to v
    g[source] = 0
    f = {v: math.inf for v in graph.vertices}  # Estimated total cost: f(v) = g(v) + h(v)
    f[source] = 0  # Heuristic h(source) = 0
    pred = {source: None}
    pq = [(0, source)]  # Priority queue: (f(v), v)
    visited = set()
    
    # Simple heuristic: h(v) = 0 (Dijkstra-like, admissible)
    def heuristic(v):
        return 0
    
    while pq:
        f_score, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        
        for v, weight in graph.get_neighbors(u):
            edge_count += 1
            if v not in visited:
                new_g = g[u] + weight
                if new_g < g[v]:
                    g[v] = new_g
                    f[v] = g[v] + heuristic(v)
                    pred[v] = u
                    heapq.heappush(pq, (f[v], v))
    
    if g[target] == math.inf:
        return [], math.inf, edge_count
    
    # Construct path
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        curr = pred[curr]
    return path[::-1], g[target], edge_count
def run_performance_test(graph, source, target, iterations=100, name="Graph"):
    print(f"\n{name} (from {source} to {target}):")
    
    nx_path, nx_length, _ = networkx_dijkstra(graph, source, target)
    print(f"NetworkX: Length={nx_length}")

    # Bidirectional
    total_bidi_time = 0
    total_bidi_edges = 0
    bidi_path = None
    bidi_length = None
    for i in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = bidirectional_dijkstra(graph, source, target)
        total_bidi_time += time.perf_counter() - start_time
        total_bidi_edges += edges
        if bidi_path is None:
            bidi_path, bidi_length = path, length
    avg_bidi_time = total_bidi_time / iterations * 1000
    avg_bidi_edges = total_bidi_edges / iterations
    print(f"Bidirectional Dijkstra: Length={bidi_length}, Path={bidi_path}")
    print(f"  Avg Time: {avg_bidi_time:.2f} ms")

    # Unidirectional
    total_dijk_time = 0
    total_dijk_edges = 0
    dijk_path = None
    dijk_length = None
    for _ in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = dijkstra(graph, source, target)
        total_dijk_time += time.perf_counter() - start_time
        total_dijk_edges += edges
        if dijk_path is None:
            dijk_path, dijk_length = path, length
    avg_dijk_time = total_dijk_time / iterations * 1000
    avg_dijk_edges = total_dijk_edges / iterations
    print(f"Unidirectional Dijkstra: Length={dijk_length}, Path={dijk_path}")
    print(f"  Avg Time: {avg_dijk_time:.2f} ms")

    # A*
    total_astar_time = 0
    total_astar_edges = 0
    astar_path = None
    astar_length = None
    for _ in range(iterations):
        start_time = time.perf_counter()
        path, length, edges = astar(graph, source, target)
        total_astar_time += time.perf_counter() - start_time
        total_astar_edges += edges
        if astar_path is None:
            astar_path, astar_length = path, length
    avg_astar_time = total_astar_time / iterations * 1000
    avg_astar_edges = total_astar_edges / iterations
    print(f"A*: Length={astar_length}, Path={astar_path}")
    print(f"  Avg Time: {avg_astar_time:.2f} ms")

    return {
        'name': name,
        'uni_time': avg_dijk_time,
        'bidi_time': avg_bidi_time,
        'astar_time': avg_astar_time,
        'uni_edges': avg_dijk_edges,
        'bidi_edges': avg_bidi_edges,
        'astar_edges': avg_astar_edges
    }

# Modified time-based trend line graph (fix overlapping labels)
def plot_comparison_time_trend(results):
    graph_names = [res['name'] for res in results]
    uni_times = [res['uni_time'] for res in results]
    bidi_times = [res['bidi_time'] for res in results]
    astar_times = [res['astar_time'] for res in results]
    
    x = np.arange(len(graph_names))
    plt.figure(figsize=(12, 6))
    plt.plot(x, uni_times, marker='o', linestyle='-', color='skyblue', label='Unidirectional')
    plt.plot(x, bidi_times, marker='s', linestyle='--', color='lightcoral', label='Bidirectional')
    plt.plot(x, astar_times, marker='^', linestyle=':', color='limegreen', label='A*')
    
    plt.xlabel('Graph Type')
    plt.ylabel('Average Execution Time (ms)')
    plt.title('Trend of Execution Times: Unidirectional vs. Bidirectional vs. A*')
    plt.xticks(x, graph_names, rotation=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Staggered labels to avoid overlap
    max_time = max(max(uni_times), max(bidi_times), max(astar_times))
    offset = 0.1 * max_time  # Increased offset for clarity
    for i, v in enumerate(uni_times):
        plt.text(i, v + 0.5*offset, f'{v:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)
    for i, v in enumerate(bidi_times):
        plt.text(i, v - 0.5*offset, f'{v:.2f}', ha='center', va='top', rotation=0, fontsize=8)
    for i, v in enumerate(astar_times):
        plt.text(i, v, f'{v:.2f}', ha='center', va='center', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comparison_times_trend_with_astar.png')
    plt.close()

# New edge-based trend line graph
def plot_comparison_edge_trend(results):
    graph_names = [res['name'] for res in results]
    uni_edges = [res['uni_edges'] for res in results]
    bidi_edges = [res['bidi_edges'] for res in results]
    astar_edges = [res['astar_edges'] for res in results]
    
    x = np.arange(len(graph_names))
    plt.figure(figsize=(12, 6))
    plt.plot(x, uni_edges, marker='o', linestyle='-', color='skyblue', label='Unidirectional')
    plt.plot(x, bidi_edges, marker='s', linestyle='--', color='lightcoral', label='Bidirectional')
    plt.plot(x, astar_edges, marker='^', linestyle=':', color='limegreen', label='A*')
    
    plt.xlabel('Graph Type')
    plt.ylabel('Average Edges Visited')
    plt.title('Trend of Edges Visited: Unidirectional vs. Bidirectional vs. A*')
    plt.xticks(x, graph_names, rotation=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Staggered labels to avoid overlap
    max_edges = max(max(uni_edges), max(bidi_edges), max(astar_edges))
    offset = 0.1 * max_edges
    for i, v in enumerate(uni_edges):
        plt.text(i, v + 0.5*offset, f'{v:.1f}', ha='center', va='bottom', rotation=0, fontsize=8)
    for i, v in enumerate(bidi_edges):
        plt.text(i, v - 0.5* offset, f'{v:.1f}', ha='center', va='top', rotation=0, fontsize=8)
    for i, v in enumerate(astar_edges):
        plt.text(i, v, f'{v:.1f}', ha='center', va='center', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comparison_edges_trend_with_astar.png')
    plt.close()

# Unified test function (modified to call edge graph)
def test_algorithms_with_plot():
    results = []
    
    g_small = Graph(directed=True)
    g_small.add_edge(1, 2, 4)
    g_small.add_edge(1, 3, 2)
    g_small.add_edge(2, 4, 3)
    g_small.add_edge(3, 2, 1)
    g_small.add_edge(3, 4, 5)
    g_small.add_edge(4, 5, 2)
    g_small.add_edge(2, 5, 10)
    results.append(run_performance_test(g_small, 1, 5, iterations=100, name="Small Directed"))

    g_medium = generate_random_graph(n=10000, m=50000, directed=True)
    results.append(run_performance_test(g_medium, 0, 9999, iterations=10, name="Medium Directed"))

    g_large = generate_random_graph(n=50000, m=250000, directed=True)
    results.append(run_performance_test(g_large, 0, 49999, iterations=10, name="Large Directed"))

    g_very_large = generate_random_graph(n=100000, m=500000, directed=True)
    results.append(run_performance_test(g_very_large, 0, 99999, iterations=10, name="Very Large Directed"))

    plot_comparison_time_trend(results)
    plot_comparison_edge_trend(results)

if __name__ == "__main__":
    test_algorithms_with_plot()