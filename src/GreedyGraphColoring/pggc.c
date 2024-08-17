// Parallel greedy graph coloring algorithm
// -------------------------------
// Performs a Greedy Graph Coloring starting from vertex 1
//
// The parallel greedy graph coloring algorithm assigns colors to all vertices in parallel, 
// with each thread using a local list to track unavailable colors for each vertex. After 
// the initial coloring, the algorithm checks for neighboring vertices with the same color 
// and stores them in a global list using prefix sum. These vertices are then recolored, and 
// the process repeats until no neighboring vertices share the same color.
//
// Parameters:
// n     : number of vertices
// ver   : ver[i] points to the start of the neighbor list of vertex i in edges
// edges : lists of neighbors of each vertex, each edge is listed in both direction
// result: array of length n used for assigned colors
// T     : array used to share the number of found vertices between threads
// S     : array used to share found vertices between threads
//
// The vertices are numbered from 1 to n.

void pggc(int n,int *ver,int *edges,int *result,int *S,int *T) {
    int i, j, k, v, w, count;

    int *local_c = malloc(n * sizeof(int)); // Local array for threads to track available colors
    int *local_S = malloc(n * sizeof(int)); // Local array for threads to track vertices that need to change color

    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    // Initialize all values in local_c as -1
    for (i = 0; i < n; i++) {
        local_c[i] = -1;
    }

    // Initialize all values in result as -1
    #pragma omp for
    for (i = 1; i <= n; i++) {
        result[i] = -1;
    }
    #pragma omp barrier

    // Assign colors to all vertices. Could end up with two neighbours with the same color
    #pragma omp for
    for (i = 1; i <= n; i++) {
        for(j = ver[i]; j < ver[i+1]; j++) { // Go through the neighbors of i
            w = edges[j]; // Neighbour of i
            if (result[w] != -1) {
                local_c[result[w]] = w;
            }
        }
        // Find first available color
        for (j = 1; j <= n; j++) {
            if (local_c[j] == -1) {
                result[i] = j;
                break;
            }
        }
        // Reset color availability based on this vertex's neighbors only
        for (j = ver[i]; j < ver[i+1]; j++) {
            w = edges[j];
            if (result[w] != -1) {
                local_c[result[w]] = -1; // Reset only used indices
            }
        }
    }
    #pragma omp barrier
    T[num_threads] = 1; // Set as 1 for the first round to get the while loop to run

    while (T[num_threads] != 0) {

        // Loop through all vertices again and check if two neighbours have the same color.
        // If they have the same color, add the parent to a separate list and mark the color as -1
        count = 0; // Count of how many vertices the thread finds that need to change color
        #pragma omp for
        for (i = 1; i <= n; i++) {
            for(j = ver[i]; j < ver[i+1]; j++) { // Go through the neighbors of i
                w = edges[j]; // Neighbour of i
                if (result[i] == result[w]) { // If the two neighbours have the same color
                    local_S[count] = i;
                    result[i] = -1;
                    count++;
                    break; // Do not need to continue checking neighbours as i is already added to found list and need to change color
                }
            }
        }

        T[thread_id] = count;

        #pragma omp barrier

        // Calculate prefix sum so that the found vertices can be divided between the threads

        // Prefix sum up to thread id
        int prefix_sum = 0;
        for (i = 0; i < thread_id; i++) {
            prefix_sum += T[i];
        }

        // Merge the local list of discovered vertices into the global list S
        for(i = 0; i < count; i++) {
            S[prefix_sum + i] = local_S[i]; // Add to the global list
        }

        // Find total number of vertices found amoung the threads
        if (thread_id == num_threads - 1) {
            prefix_sum += count;
            T[num_threads] = prefix_sum;
        }

        #pragma omp barrier

        // Assign colors to all found vertices.
        #pragma omp for
        for (i = 0; i < T[num_threads]; i++) {
            v = S[i];
            for(j = ver[v]; j < ver[v+1]; j++) { // Go through the neighbors of v
                w = edges[j]; // Neighbour of v
                local_c[result[w]] = w;
            }
            // Find first available color
            for (j = 1; j <= n; j++) {
                if (local_c[j] == -1) {
                    result[v] = j;
                    break;
                }
            }
            // Reset color availability based on this vertex's neighbors only
            for (j = ver[v]; j < ver[v+1]; j++) {
                w = edges[j];
                if (result[w] != -1) {
                    local_c[result[w]] = -1; // Reset only used indices
                }
            }
        }

        #pragma omp single
        T[num_threads] = 0; // Assign 0 value, and if it is still 0 after check below, then the coloring is succesful

        #pragma omp barrier

        // Check that all vertices have a different color to its neighbour, otherwise go through the while loop again
        #pragma omp for
        for (i = 1; i <= n; i++) {
            if (T[num_threads] == 0) {
                for(j = ver[i]; j < ver[i+1]; j++) { // Go through the neighbors of i
                    w = edges[j]; // Neighbour of i
                    if (result[i] == result[w]) { // If the two neighbours have the same color
                        #pragma omp critical
                        T[num_threads] = 1; // Do not need to continue checking neighbours because coloring was not succesful and we need to run the while loop again
                        break;
                    }
                }
            }
        }
        #pragma omp barrier
    }
}