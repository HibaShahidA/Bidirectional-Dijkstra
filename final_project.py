
from collections import defaultdict
import heapq
import math
import time
import networkx as nx
import random


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
    print(f"  Avg Time: {avg_bidi_time:.3f} ms, Avg Edges: {avg_bidi_edges:.1f}")

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
    print(f"  Avg Time: {avg_dijk_time:.3f} ms, Avg Edges: {avg_dijk_edges:.1f}")

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

if __name__ == "__main__":
    test_algorithms()