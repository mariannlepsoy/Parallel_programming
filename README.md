# Parallel Programming

## Overview
This repository contains parallel implementations of classic algorithms, all written in C and utilizing OpenMP for parallelization. The repository includes two parallel versions of the Breadth-First Search (BFS) algorithm and one parallel implementation of the greedy graph coloring algorithm.

## Algorithms

### [Parallel BFS](https://github.com/mariannlepsoy/Parallel_programming/blob/main/src/BFS/pbfs.c)  
A parallelized BFS algorithm that distributes the workload of exploring vertices across multiple threads, optimizing for large graphs. After each round `i`, the algorithm gathers all discovered vertices and redistributes them among the threads for the search in the next round, `i+1`. This approach ensures balanced distribution of the workload across threads during the parallel execution.


### [Alternativ Parallel BFS](https://github.com/mariannlepsoy/Parallel_programming/blob/main/src/BFS/abfs.c)
This algorithm begins with several rounds of sequential BFS before transitioning to parallel execution. During the parallel phase, each thread is allocated a portion of the vertices discovered in the last sequential round. Any newly discovered vertices in the parallel portion remain with the thread that found them, continuing until the graph has been fully explored.

### [Parallel Greedy Graph Coloring](https://github.com/mariannlepsoy/Parallel_programming/blob/main/src/GreedyGraphColoring/pggc.c)
The parallel greedy graph coloring algorithm assigns colors to all vertices in parallel, with each thread using a local list to track unavailable colors for each vertex. After the initial coloring, the algorithm checks for neighboring vertices with the same color and stores them in a global list using prefix sum. These vertices are then recolored, and the process repeats until no neighboring vertices share the same color.