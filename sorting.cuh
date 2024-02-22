#ifndef SORTING_CUH
#define SORTING_CUH


typedef struct {
    size_t len;
    size_t start;
    size_t end;
}block_info;


/**
 * Kernel implementing the step n. 1 of the algorithm
*/
__global__ void bitonic_sort(int* x, int sequence_size);

/**
 * Kernel implementing the step n. 2 of the algorithm
*/
__global__ void merge(int *x, int *out, int sequence_size);

/**
 * Kernel implementing the step n. 4 of the algorithm
*/
__global__ void final_merge(int **ins, int **outs, block_info** b_infos, const int L);

#endif