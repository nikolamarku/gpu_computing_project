#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include "sorting.cuh"


/**
 * Phase implementation of a bitonic sort network
*/
__device__ void phase(int *x, int sequence_size, int comparator_size, bool two_ways, bool full){
    assert(blockDim.x == 32);
    int MAX_COMPARATOR_N = (sequence_size/2)/2;
    int idx = threadIdx.x;
    int num_of_comparators = (sequence_size / comparator_size) /2;
    int selected_comparator = idx / (MAX_COMPARATOR_N/num_of_comparators);
    __shared__ int tile[128];

    //load data in shared mem
    tile[idx] = x[idx];
	tile[idx + 32] = x[idx + 32];
    if(full){
	    tile[idx + 64] = x[idx + 64];
	    tile[idx + 96] = x[idx + 96];
    }

    //run all stages of this phase
    for (int j = comparator_size / 2; j > 0; j /= 2) {
        int groups_in_comp_stage = comparator_size / (j*2);
        int selected_group = idx % groups_in_comp_stage;
        int group_start = selected_group * (2*j);
        int offset_in_group = (idx/groups_in_comp_stage) % j;
        int k0 = (selected_comparator * (comparator_size*2)) + group_start + offset_in_group;
        int k1 = sequence_size - 1 - k0;
        //ascending compare and swap
        int a = tile[k0], b = tile[k0+j];
        tile[k0] =   min(a,b);
        tile[k0+j] = max(a,b);
        if(full){
            //descending compare and swap 
            int c= tile[k1-j], d = tile[k1];
            tile[k1-j] = (1-two_ways)*min(c,d) + two_ways*max(c,d);
            tile[k1] =   (1-two_ways)*max(c,d) + two_ways*min(c,d);
        }
    }

    //load result in global mem
    x[idx] = tile[idx];
	x[idx + 32] = tile[idx + 32];
    if(full){
        x[idx + 64] = tile[idx + 64];
        x[idx + 96] = tile[idx + 96];
    }
}

__global__ void bitonic_sort(int *x, int sequence_size){
    assert(blockDim.x == (sequence_size/4));
    int offset = blockIdx.x * sequence_size;
    //run all phases of the network (except the last)
    for (int i = 2; i < sequence_size; i *= 2)
        phase(x + offset,sequence_size,i,true,true);

    //run the last phase (merge the bitonic sequences)
    phase(x + offset,sequence_size,sequence_size,false,true);
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
    //load data in smem such that the two subsequences form a bitonic sequence (descending to ascending)
    tile[32 - 1 - idx] =  A[idx];
	tile[32 + idx]     =  B[idx];
    int max_A = tile[0], max_B = tile[63];
    int A_cursor = 32, B_cursor = 32;

    //while there are subsequences in A or B to merge
    while(A_cursor < sequence_size || B_cursor < sequence_size ){
      //merge the bitonic sequence
      phase(tile,64,64,false,false);
      //load in global mem the first half
      out[ A_cursor + B_cursor - 64 + idx] = tile[idx];
      //fill the first half
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
    //merge
    phase(tile,64,64,false,false);
    //load result in mem
    out[A_cursor + B_cursor - 64 + idx] = tile[idx];
    out[A_cursor + B_cursor - 64 + idx + 32 ] = tile[idx + 32];
}



__global__ void final_merge(int **ins, int **outs, block_info** b_infos, const int L){
    assert(blockDim.x == 32);
    int s = blockIdx.x;
    int i = blockIdx.y * (L/gridDim.y);
    int k = (L/2) / gridDim.y;
    int idx = threadIdx.x;

    int offset = b_infos[s][i].start,
        seq_a_size = b_infos[s][i].len,
        seq_a_pos = b_infos[s][i].start,
        seq_b_size = b_infos[s][i+k].len, 
        seq_b_pos = b_infos[s][i+k].start;
    int *out = outs[s] + offset;

    int *A = ins[s] + seq_a_pos, *B = ins[s] + seq_b_pos;
    __shared__ int tile[64];
    //load data in smem such that the two sequences form a bitonic sequence (desc to asc).
    tile[32 - 1 - idx] =  (idx < seq_a_size) ? A[idx] : INT_MAX;
	tile[32 + idx]     =  (idx < seq_b_size) ? B[idx] : INT_MAX;
    int max_A = tile[0], max_B = tile[63];
    int A_cursor = min(seq_a_size,32);
    int B_cursor = min(seq_b_size, 32);
    int copied = 0;

    //while there are subsequences in A or B to merge
    while(A_cursor < seq_a_size || B_cursor < seq_b_size ){
        //merge the bitonic sequence
        phase(tile,64,64,false,false);
        //load in global mem the first half
        out[ copied + idx ] = tile[idx];
        copied+=32;
        //fill the first half
        if((max_A <= max_B && A_cursor < seq_a_size) || B_cursor == seq_b_size){
            assert(A_cursor < seq_a_size);
            tile[32 - 1 - idx] =  ((idx+A_cursor) < seq_a_size) ? A[idx + A_cursor] : INT_MAX;
            max_A = tile[0];
            A_cursor = min(A_cursor + 32, seq_a_size);
        }else{
            assert(B_cursor < seq_b_size);
            tile[32 - 1 - idx] = ((idx+B_cursor) < seq_b_size) ? B[idx + B_cursor] : INT_MAX;
            max_B = tile[0];
            B_cursor = min(B_cursor + 32, seq_b_size);
        }
    }
    //merge
    phase(tile,64,64,false,false);

    //check if it fits output
    if(tile[idx] < INT_MAX)
        out[copied + idx] = tile[idx];
    if(tile[idx+32] < INT_MAX)
        out[copied + idx + 32] = tile[idx + 32];

    //update block info
    b_infos[s][i].end = b_infos[s][i+k].end;
    b_infos[s][i].len = b_infos[s][i].len + b_infos[s][i+k].len;
}