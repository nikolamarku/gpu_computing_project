#include <stdio.h>
#include "sorting.cuh"
#include <stdlib.h>
#include <time.h>

#define S 64
#define K 4

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
    return splitters;
}

int main(){
    srand(time(NULL));
    int sequence_size = 64;
    int n = sequence_size * 2;
    int size = sizeof(int) * n;
    int *v = (int*) malloc(size);
    for(int i=0; i<n; i++) v[i] = rand() % 1000;
    int *splitters = get_splitters(v,n);

    int *d_in, *d_out;

    cudaMalloc(&d_in, size) ;
    cudaMemcpy(d_in, v, size, cudaMemcpyHostToDevice);
    bitonic_sort<<<n/sequence_size,sequence_size/4>>>(d_in,sequence_size);
    cudaDeviceSynchronize();
    cudaMemcpy(v, d_in, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++)
        printf("%2d ",v[i]);

    printf("\n");

    cudaMalloc(&d_out, size) ;
    for(int seq = sequence_size; seq< n; seq *=2){
        merge<<<n/(seq*2),32>>>(d_in,d_out,seq);
        cudaMemcpy(d_in, d_out, size, cudaMemcpyDeviceToDevice);
    }
   
    cudaDeviceSynchronize(); 
    cudaMemcpy(v, d_out, size, cudaMemcpyDeviceToHost);
    

    for(int i = 0; i < n; i++)
        printf("%2d ",v[i]);

    printf("\n");

    
    return 0;
}
