__global__ void bitonic_sort(int* x, int sequence_size);
__global__ void merge(int *x, int *out, int sequence_size);
__global__ void arb_merge(int *x, int *out, int offset, int seq_a_size, int seq_a_pos, int seq_b_size, int seq_b_pos);