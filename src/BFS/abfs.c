
// ALternative Parallel Breadth First Search
// -----------------------------
// Berforms a BFS starting from vertex 1
// The parent of each vertex in the BFS tree along with its distance from the starting
// vertex is computed.
//
// The algorithm should first perform some rounds of sequential BFS before starting a parallel
// execution. In the parallel part each thread should be allocated a part of the vertices from the
// last round of the sequential algorithm. Any discovered vertices in the parallel part should 
// remain with the thread that discovered them. This continues until the entire graph has been
// explored.
//
//
// Parameters:
// n     : number of vertices
// ver   : ver[i] points to the start of the neighbor list of vertex i in edges
// edges : lists of neighbors of each vertex, each edge is listed in both direction
// p     : array of length n used for parent pointers
// dist  : array of length n used for distance from starting vertex
// S     : array of length n used for maintaining queue of vertices to be processed, only used in the 
//         sequential part. 
// T     : array of length n where n >> number of threads. 
//
// Note that the vertices are numbered from 1 to n (inclusive). Thus there is
// no vertex 0.

void abfs(int n,int *ver,int *edges,int *p,int *dist,int *S,int *T) {

    int i,j,r;          // Loop indices
    int v,w;          // Pointers to vertices
    int k = 2;
    int *temp;        // Temporary pointer
    
    #pragma omp for
    for(i=1;i<=n;i++) {   // Set that every node is unvisited
    p[i] = -1;          // Using -1 to mark that a vertex is unvisited
    dist[i] = -1;
    }

    p[1] = 1;        // Set the parent of starting vertex to itself
    dist[1] = 0;     // Set the distance from the starting vertex to itself
    S[0] = 1;        // Start vertex

    int *local_S = malloc(n * sizeof(int)); // Thread-local array for newly discovered vertices
    int *local_T = malloc(n * sizeof(int));
    int local_num_w = 0; // Number of vertices discovered locally
    int local_num_r; // Number of vertices to search locally

    int thread_id = omp_get_thread_num(); // Thread id
    int num_threads = omp_get_num_threads(); // Number of threads

    T[num_threads] = 1;        // Number of vertices to start with.

    while (T[num_threads] != 0) {

        // Divide S between the threads
        local_num_r = 0;
        #pragma omp for
        for (i=0;i<T[num_threads];i++) {
            local_S[local_num_r++] = S[i];
        }

        #pragma omp barrier

        for (r=0;r<k;r++) {       // Run sequentially for first k rounds, after k rounds each thread will work on own local array
            #pragma omp barrier   // Ensure all threads are at the same level

            for(i=0;i<local_num_r;i++) {           // Loop over vertices in S
                v = local_S[i];                      // Grab next vertex v in S
                for(j=ver[v];j<ver[v+1];j++) { // Go through the neighbors of v
                    w = edges[j];                // Get next neighbor w of v
                    if (p[w] == -1) {            // Check if w is undiscovered
                        p[w] = v;                  // Set v as the parent of w
                        dist[w] = dist[v]+1;       // Set distance of w 
                        local_T[local_num_w++] = w;            // Add w to T and increase number of vertices discovered 
                    }
                }  // End loop over neighbors of v
            }  // End loop of vertices in S
            temp = local_S; // Swap local_S and local_T
            local_S = local_T;
            local_T = temp;
            local_num_r = local_num_w; // Set the number of vertices to search locally next iteration to the number discovered this iteration
            T[thread_id] = local_num_w; // Share how many children each thread found in global list
            local_num_w = 0; // Reset local_num_w
        }
        // Synchronization barrier to ensure all threads have processed their vertices for k iterations
        #pragma omp barrier

        // Prefix sum up to thread id
        int prefix_sum = 0;
        for (i = 0; i < thread_id; i++) {
            prefix_sum += T[i];
        }
        
        for(i = 0; i < T[thread_id]; i++) {
            S[prefix_sum + i] = local_S[i]; // Add to the global list
        }

        if (thread_id == num_threads - 1) {
            prefix_sum += T[thread_id];
            T[num_threads] = prefix_sum;
        }
        #pragma omp barrier
    }
    free(local_S);
    free(local_T);
}
