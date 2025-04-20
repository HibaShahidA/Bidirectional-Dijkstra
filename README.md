# Bidirectional-Dijkstra
Applying a bidirectional Dijkstra algorithm to prove that it is instance-optimal. This is done under the use of the research paper titles: "Bidirectional Dijkstra's Algorithm is Instance-Optimal", by authors: Bernhard Haeupler, RIchard Hladik, Vaclav Rozhon, Robert E. Tarjan, akub Tetek.

# Algorithmic details
The algorithm runs like the standard Dijkstra, except in the _bidirectional search_, the search starts from both the endpoints - the source vertex _s_ and the target vertex _t_. This approach can be imagined as two people walking towards one another, taking the optimal steps and meeting in the middle. Not only did both the people take the shortest routes to the middle, but they also did not need to cross all the way to the other side. Similarly, in the bidirectional search, not all nodes need to be traversed, hence reducing the nodes that need to be searched.

# Dataset
Since _bidirectional search_ shows promising results in huge graphs, it is only fair to run the algorithm against the standard Dijkstra on equally large datasets. Therefore, we have decided to use two approaches for the final testing:
- _synthetic_ datasets of varying sizes to test the optimal time on expected output
- _code generated_ datasets of varying sizes and types

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
