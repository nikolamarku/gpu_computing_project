#ifndef SORTING_CUH
#define SORTING_CUH


typedef struct {
    int len;
    int start;
    int end;
}block_info;

__global__ void bitonic_sort(int* x, int sequence_size);
__global__ void merge(int *x, int *out, int sequence_size);
__global__ void final_merge(int *x, int *out, block_info *block_info, const int L);

#endif