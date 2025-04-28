# Bidirectional-Dijkstra

## Overview
Applying a bidirectional Dijkstra algorithm to prove that it is instance-optimal. This is done under the use of the research paper titles: "Bidirectional Dijkstra's Algorithm is Instance-Optimal", by authors: Bernhard Haeupler, RIchard Hladik, Vaclav Rozhon, Robert E. Tarjan, akub Tetek.

## Algorithmic details
The algorithm runs like the standard Dijkstra, except in the _bidirectional search_, the search starts from both the endpoints - the source vertex _s_ and the target vertex _t_. This approach can be imagined as two people walking towards one another, taking the optimal steps and meeting in the middle. Not only did both the people take the shortest routes to the middle, but they also did not need to cross all the way to the other side. Similarly, in the bidirectional search, not all nodes need to be traversed, hence reducing the nodes that need to be searched.

## Key Features
- **Instance-optimality**: The algorithm is designed to achieve optimal performance based on the specific instance of the graph.
- **Bidirectional search**: Improves efficiency by reducing the number of nodes traversed.
- **Scalability**: Particularly efficient for large graphs with many vertices and edges.

## Dataset
Since **bidirectional search** is particularly useful for large graphs, we test the algorithm against the standard Dijkstra algorithm on equally large datasets to compare performance.

For final testing, we used two different approaches:
1. **Synthetic datasets** of varying sizes to test the algorithm's performance against expected outputs.
2. **Code-generated datasets** of varying sizes and types to test its adaptability and efficiency across different graph structures.

## Results
The results of the performance comparison between the Bidirectional Dijkstra algorithm and standard Dijkstra are provided in the `benchmarks` directory. The performance is evaluated based on factors like:
- **Edge traversal count** (comparison_edges_trend.png)
- **Execution time** (comparison_times_trend.png)

## Directory Breakdown

- `Checkpoint1/`
  - `BidirectionalDijkstra.pdf`
  - `BidirectionalDijkstra.tex`
- `Checkpoint2/`
  - `BidirectionalDijkstra-cp2.pdf`
  - `report.tex`
- `Checkpoint3/`
  - `BidirectionalDijkstra-cp3.pdf`
  - `report(3).tex`
- `Checkpoint4/`
  - `Bidirectional Dijkstra Optimizing Shortest Paths.pdf`
  - `FinalReport.pdf`
  - `final_report.tex`
- `benchmarks/`
  - `comparison_edges_trend.png`
  - `comparison_times_trend.png`
  - `performance.txt`
- `data`
  - `data_gen.py`
  - `graph_datasets.zip`
- `docs`
  - `BidirectionalDijkstra-cp2-edit.jpg`
  - `notes.txt`
- `src`
  - `main.py`

## How to Run

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Bidirectional-Dijkstra
```

### 2. Install Dependencies
``` bash
pip install networkx matplotlib numpy
```

### 3. Run the Project
```bash
python src/main.py
```
