#include <stdio.h>
#include "sorting.cuh"
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <assert.h>
#include <time.h>

#define SEQUENCE_SIZE 128
#define S 128
#define K 64
#define L 64
#define CEIL(x,n) (x/n)*n + (n * (x % n > 0))
#define CHECK_PTR(ptr) {\
    if(ptr == NULL){ \
        printf("ptr is null %s:%d\n",__FILE__,__LINE__); \
        exit(1); \
    } \
}

#define CHECK(x) { \
    x;                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

bool is_power_of_2(int x){
    return (x != 0) && ((x & (x - 1)) == 0);
}

int cmp(const void * a, const void * b){
    return *((int*)a) - *((int*)b);
}


int *get_splitters(int *in, int n){
    int *tmp = (int*) malloc(sizeof(int)*S*K);
    int *splitters = (int*) malloc(sizeof(int)*S);
    for(int i = 0; i < S*K; i++)
        tmp[i] = in[rand() % n];
    qsort(tmp,S*K,sizeof(int),cmp);
    for(int i=0; i < S; i++)
        splitters[i] = tmp[i*K];

    splitters[S-1] = INT_MAX;
    return splitters;
}



void warpSort(int *in, int n){
    assert(n % 64 == 0 && is_power_of_2(n));
    int size = sizeof(int) * n;
    int *out = (int*) malloc(size);

    int *splitters = get_splitters(in,n);
    int *d_in, *d_out;

    //step 1: bitonic sort
    CHECK(cudaMalloc(&d_in, size));
    CHECK(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    bitonic_sort<<<n/SEQUENCE_SIZE,SEQUENCE_SIZE/4>>>(d_in,SEQUENCE_SIZE);
    CHECK(cudaMemcpy(out, d_in, size, cudaMemcpyDeviceToHost));

    //step 2: merge
    CHECK(cudaMalloc(&d_out, size));
    for(int seq = SEQUENCE_SIZE; (n/seq) > L; seq *=2){
        merge<<<n/(seq*2),32>>>(d_in,d_out,seq);
        CHECK(cudaMemcpy(d_in, d_out, size, cudaMemcpyDeviceToDevice));
    }
    CHECK(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_out));


    //step 3: split into small tiles
    block_info **block_len = (block_info**) malloc(sizeof(block_info*) * S);
    int blocks_len[S] = {0};
    for(int i = 0; i < S; i++)
        block_len[i] = (block_info*) calloc(L, sizeof(block_info));

    for(int i = 0; i<L; i++){
        int k = 0;
        for(int j = 0; j<S; j++){
            int start = k;
            while( k < (n/L) && out[(i*(n/L)) + k] <= splitters[j]) k++;
            block_len[j][i].start = (i*(n/L)) + start;
            block_len[j][i].end = (i*(n/L)) + k;
            block_len[j][i].len = k - start;
            blocks_len[j] += (k -start);
        }
    }
    //re arrange
    int *organized_input = (int *) calloc(n, sizeof(int));
    int z = 0;
    for(int j=0; j<S; j++){
        int start = 0;
        for(int i=0; i<L; i++){
            int a = z;
            for(int k=block_len[j][i].start; k < block_len[j][i].end; k++)
                organized_input[z++] = out[k];
            block_len[j][i].start = start;
            block_len[j][i].end = start + z - a;
            start = start + z - a;
        }
    }

    int **h_ins;
    int **h_outs;
    h_ins = (int**) malloc(sizeof(int*) * S);
    h_outs = (int**) malloc(sizeof(int*) * S);
    int offset = 0;
    for(int i =0; i<S; i++){
        CHECK(cudaMalloc(&h_ins[i],sizeof(int)*blocks_len[i]));
        CHECK(cudaMemcpy(h_ins[i],organized_input + offset,sizeof(int) * blocks_len[i],cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&h_outs[i],sizeof(int)*blocks_len[i]));
        offset += blocks_len[i];
    }

    block_info **h_block_len = (block_info**) malloc(sizeof(block_info*) * S);
    for(int i = 0; i < S; i++){
        CHECK(cudaMalloc(&h_block_len[i],sizeof(block_info) * L));
        CHECK(cudaMemcpy(h_block_len[i],block_len[i],sizeof(block_info) * L,cudaMemcpyHostToDevice));
    }


    //step 4: merge independent S sequences
    int **d_ins;
    int **d_outs;
    block_info **dd_block_info;
    cudaMalloc(&d_ins, sizeof(int*)*S);
    cudaMalloc(&d_outs, sizeof(int*)*S);
    cudaMalloc(&dd_block_info, sizeof(block_info*)*S);
    
    cudaMemcpy(d_ins, h_ins, sizeof(int*)*S,cudaMemcpyHostToDevice);
    cudaMemcpy(d_outs, h_outs, sizeof(int*)*S, cudaMemcpyHostToDevice);
    cudaMemcpy(dd_block_info, h_block_len, sizeof(int*)*S, cudaMemcpyHostToDevice);

    for(int k = L/2; k > 0; k /= 2){
        dim3 grid(S,k);
        final_merge<<<grid,32>>>(d_ins,d_outs,dd_block_info,L);
        CHECK();
        for(int i = 0; i < S; i++)
            CHECK(cudaMemcpy(h_ins[i],h_outs[i],blocks_len[i]*sizeof(int),cudaMemcpyDeviceToDevice));
    }

    offset = 0;
    for(int s = 0; s < S; s++){
        CHECK(cudaMemcpy(out + offset,h_outs[s],sizeof(int) * blocks_len[s],cudaMemcpyDeviceToHost));
        offset += blocks_len[s];
    }
    cudaDeviceSynchronize();
    memcpy(in,out,sizeof(int)*n);
}

int main(int argc, char **argv){
    srand(time(NULL));
    int n = SEQUENCE_SIZE * (1 << atoi(argv[1]));
    int size = sizeof(int) * n;
    int *in = (int*) malloc(size);
    CHECK_PTR(in);
    for(int i=0; i<n; i++) in[i] = rand() % 10000;
    int *in2 = (int*) malloc(size);
    CHECK_PTR(in2);
    memcpy(in2,in,sizeof(int)*n); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    warpSort(in,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float warpsort_milliseconds = 0;
    cudaEventElapsedTime(&warpsort_milliseconds, start, stop);

    clock_t begin = clock();
    qsort(in2,n,sizeof(int),cmp);
    clock_t end = clock();
    double qsort_seconds = (double)(end - begin) / CLOCKS_PER_SEC;

    assert(memcmp(in,in2,sizeof(int)*n) == 0);
    printf("%d;%f;%f\n",n,warpsort_milliseconds/1000,qsort_seconds);
    return 0;
}
