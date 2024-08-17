// Parallel Breadth First Search
// -----------------------------
// Performs a BFS starting from vertex 1
// The parent of each vertex in the BFS tree along with its distance from the starting
// vertex is computed.
//
// The algorithm gathers all discovered vertices from round i, so that they can be 
// distributed among the threads before the search in round i+1.
//
// Parameters:
// n     : number of vertices
// ver   : array of length n. ver[i] points to the start of the neighbor list of vertex i in edges
// edges : array containing lists of neighbors for each vertex, each edge is listed in both direction
// p     : array of length n used for parent pointers
// dist  : array of length n used for distance from starting vertex
// S     : array of length n used for maintaining queue of vertices to be processed 
// T     : array of length n where n >> number of threads. 
//
// Note that the vertices are numbered from 1 to n (inclusive). Thus there is
// no vertex 0.

void pbfs(int n,int *ver,int *edges,int *p,int *dist,int *S,int *T) {

    int i,j;          // Loop indices
    int v,w;          // Pointers to vertices

    #pragma omp for
    for(i=1;i<=n;i++) {   // Set that every node is unvisited
    p[i] = -1;          // Using -1 to mark that a vertex is unvisited
    dist[i] = -1;
    }

    p[1] = 1;        // Set the parent of starting vertex to itself
    dist[1] = 0;     // Set the distance from the starting vertex to itself
    S[0] = 1;        // Add the starting vertex to S

    int *local_S = malloc(n * sizeof(int)); // Thread-local array for newly discovered vertices
    int local_num_w = 0; // Number of vertices discovered locally

    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    // T[num_threads] is how many vertices to check for the next level
    T[num_threads] = 1; // Starts with one

    while (T[num_threads] != 0) {               // Loop until all vertices have been discovered

    #pragma omp for
    for(i = 0; i < T[num_threads]; i++) {           // Loop over vertices in S
        v = S[i];
        for(j=ver[v];j<ver[v+1];j++) { // Go through the neighbors of v
            w = edges[j];                // Get next neighbor w of v
            if (p[w] == -1 ) {
                p[w] = v; // Mark the current vertex as visited and store its parent
                dist[w] = dist[v] + 1; // Update the distance
                local_S[local_num_w++] = w; // Add to thread's local list
            }
        }  // End loop over neighbors of v
    }  // End loop of vertices in S
    
    T[thread_id] = local_num_w; // Share how many children each thread found in global list

    // Synchronization barrier to ensure all threads have processed their vertices
    #pragma omp barrier

    // Prefix sum up to thread id
    int prefix_sum = 0;
    for (i = 0; i < thread_id; i++) {
        prefix_sum += T[i];
    }

    // Merge the local list of discovered vertices into the global list S
    for(i = 0; i < local_num_w; i++) {
        S[prefix_sum + i] = local_S[i]; // Add to the global list
    }

    if (thread_id == num_threads - 1) {
        prefix_sum += local_num_w;
        T[num_threads] = prefix_sum;
    }

    local_num_w = 0; // Reset local index after merging

    // Synchronization barrier to ensure the single thread has finished swapping the queues
    #pragma omp barrier
    }
    free(local_S);
}