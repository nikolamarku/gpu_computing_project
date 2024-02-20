#include <stdio.h>
#include "sorting.cuh"
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <assert.h>

#define S 64
#define K 64
#define L 64
#define CEIL(x,n) (x/n)*n + (n * (x % n > 0))

int cmp(const void * a, const void * b){
    return *((int*)a) - *((int*)b);
}

void test_arb_merge(){
    for(int y = 0; y < 100; y++){
        int arr_size = (rand() % 2046);
        int *d_in, *d_out;
        int *myvect= (int*) malloc(sizeof(int)*arr_size);
        int a_size = (rand() % arr_size) + 1;
        printf("arr_size: %d, a_size: %d\n",arr_size, a_size);

        for(int i=0;i<a_size;i++) myvect[i] = i;
        for(int i=a_size;i<arr_size;i++) myvect[i] = i - a_size;
        int *tocmp = (int*) malloc(sizeof(int)*arr_size);
        memcpy(tocmp,myvect,sizeof(int)*arr_size);
        
        cudaMalloc(&d_out, sizeof(int) * arr_size);
        cudaMalloc(&d_in, sizeof(int) * arr_size);
        cudaMemcpy(d_in,myvect,sizeof(int)*arr_size,cudaMemcpyHostToDevice);
        arb_merge<<<1,32>>>(d_in,d_out,0,a_size,0,arr_size - a_size,a_size);
        cudaDeviceSynchronize();
        int *o = (int*) malloc(sizeof(int)*arr_size);
        cudaMemcpy(o,d_out,sizeof(int)*arr_size,cudaMemcpyDeviceToHost);
        
        qsort(tocmp,arr_size, sizeof(int),cmp); 
        if(memcmp(tocmp,o,sizeof(int)*arr_size) != 0)
            printf("diversi\n");
    }
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
    int n = sequence_size * 128;
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

    //for(int i=0; i<S; i++ ) printf("%d ",splitters[i]);


//    for(int i = 0; i < n; i++)
//        printf("%2d ",v[i]);
//
//    printf("\n\n\n");

    cudaMalloc(&d_out, size) ;
    for(int seq = sequence_size; (n/seq) > L; seq *=2){
        merge<<<n/(seq*2),32>>>(d_in,d_out,seq);
        cudaMemcpy(d_in, d_out, size, cudaMemcpyDeviceToDevice);
    }


    //int *collector =  (int*) malloc(sizeof(int) * L * S);
    /*for(int i=0; i < L*S; i++) collector[i] = INT_MIN;*/
   
    cudaDeviceSynchronize(); 
    cudaMemcpy(v, d_out, size, cudaMemcpyDeviceToHost);
    
    /*for(int i = 0; i < L; i++){
        for(int j=0; j < (n/L); j++){
            printf("%d ", v[i*(n/L) + j]);
        }        
        printf("\n\n");
    }*/
/*
    for(int i = 0; i < n; i++)
        printf("%2d ",v[i]);

    printf("\n");
*/
    test_arb_merge();
    return 0;
}
