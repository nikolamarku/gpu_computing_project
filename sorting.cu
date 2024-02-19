#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

__device__ void phase(int *x, int sequence_size, int comparator_size, bool two_ways, bool full){
    int MAX_COMPARATOR_N = (sequence_size/2)/2;
    int idx = threadIdx.x;
    int num_of_comparators = (sequence_size / comparator_size) /2;
    //fix this
    int selected_comparator = idx / (MAX_COMPARATOR_N/num_of_comparators);

    for (int j = comparator_size / 2; j > 0; j /= 2) {
        int groups_in_comp_stage = comparator_size / (j*2);
        int selected_group = idx % groups_in_comp_stage;
        int group_start = selected_group * (2*j);
        int offset_in_group = (idx/groups_in_comp_stage) % j;
        int k0 = (selected_comparator * (comparator_size*2)) + group_start + offset_in_group;
        int k1 = sequence_size - 1 - k0;
        int a = x[k0], b = x[k0+j];
        x[k0] =   min(a,b);
        x[k0+j] = max(a,b);
        if(full){
          int c= x[k1-j], d = x[k1];
          x[k1-j] = (1-two_ways)*min(c,d) + two_ways*max(c,d);
          x[k1] =   (1-two_ways)*max(c,d) + two_ways*min(c,d);
        }
    }
}

__global__ void bitonic_sort(int *x, int sequence_size){
    assert(blockDim.x == (sequence_size/4));
    int offset = blockIdx.x * sequence_size;
    int *vect = x + offset;
    for (int i = 2; i < sequence_size; i *= 2)
        phase(vect,sequence_size,i,true,true);
    phase(vect,sequence_size,sequence_size,false,true);
}

__global__ void merge(int *x, int *out, int sequence_size){
    assert(blockDim.x == 32);
    assert(sequence_size % 32 == 0 && sequence_size >= 64);
    int offset = blockIdx.x * (sequence_size*2);
    int idx = threadIdx.x;
    int *vect = x + offset;
    out = out + offset;
    int *A = vect, *B = vect + sequence_size;
    __shared__ int tile[64];
    tile[32 - 1 - idx] =  A[idx];
	  tile[32 + idx]     =  B[idx];
    int max_A = tile[0], max_B = tile[63];
    int A_cursor = 32, B_cursor = 32;

    while(A_cursor < sequence_size || B_cursor < sequence_size ){
      phase(tile,64,64,false,false);
      out[ A_cursor + B_cursor - 64 + idx] = tile[idx];
      if((max_A <= max_B && A_cursor < sequence_size) || B_cursor == sequence_size){
        assert(A_cursor < sequence_size);
        tile[32 - 1 - idx] =  A[idx + A_cursor];
        max_A = tile[0];
        A_cursor += 32;
      }else{
        assert(B_cursor < sequence_size);
        tile[32 - 1 - idx] = B[idx + B_cursor];
        max_B = tile[0];
        B_cursor += 32;
      }
    }

    phase(tile,64,64,false,false);
    out[A_cursor + B_cursor - 64 + idx] = tile[idx];
    out[A_cursor + B_cursor - 64 + idx + 32 ] = tile[idx + 32];
}