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

typedef struct{
    int len;
    int start;
    int end;
}block_info;

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

    splitters[S-1] = INT_MAX;
    return splitters;
}

int main(){
    srand(time(NULL));
    int sequence_size = 64;
    int n = sequence_size * 128;
    int size = sizeof(int) * n;
    int *in = (int*) malloc(size);
    int *out = (int*) malloc(size);
    for(int i=0; i<n; i++) in[i] = rand() % 1000;
    int *splitters = get_splitters(in,n);
    int *d_in, *d_out;

    cudaMalloc(&d_in, size) ;
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    bitonic_sort<<<n/sequence_size,sequence_size/4>>>(d_in,sequence_size);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_in, size, cudaMemcpyDeviceToHost);

    for(int i=0; i<(n/sequence_size); i++){
        for(int j=0; j < sequence_size - 1; j++)
            assert(out[(i*sequence_size) + j] <= out[(i*sequence_size) +j +1]);
    }

    cudaMalloc(&d_out, size) ;
    for(int seq = sequence_size; (n/seq) > L; seq *=2){
        merge<<<n/(seq*2),32>>>(d_in,d_out,seq);
        cudaMemcpy(d_in, d_out, size, cudaMemcpyDeviceToDevice);
    }
   
    cudaDeviceSynchronize(); 
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);


    int *splitters_indexes = (int*) malloc(sizeof(int) * S);

    printf("splitters: ");
    for(int i=0; i  < S; i++){
        printf("%d ",splitters[i]);
    }
    printf("\n");

    // INIZIALIZZAZIONE BLOCCHI
    block_info **block_len = (block_info**) malloc(sizeof(block_info*) * L);
    for(int i = 0; i < L; i++)
        block_len[i] = (block_info*) calloc(S, sizeof(block_info));

    for(int i = 0; i<L; i++){
        int k = 0;
        for(int j = 0; j<S; j++){
            int start = k;
            while( k < (n/L) && out[(i*(n/L)) + k] <= splitters[j]) k++;
            block_len[i][j].start = (i*(n/L)) + start;
            block_len[i][j].end = (i*(n/L)) + k;
            block_len[i][j].len = k - start;
        }
    }
    
    for(int i=0; i<L; i++){
        int sum = 0;
        for(int j=0; j<S; j++)
            sum += block_len[i][j].len;
        assert(sum == (n/L));
    }

    int blocks_len[S] = {0};
    for(int s=0; s<S; s++){
        int sum = 0;
        for(int l=0; l<L; l++){
            sum += block_len[l][s].len;
        }
        blocks_len[s] = sum;
    }

    // FINE INIZIALIZZAZIONE BLOCCHI

    cudaMemset( d_out, 0,sizeof(int)*n);

    int *organized_input = (int *) calloc(n, sizeof(int));
    int z = 0;
    for(int j=0; j<S; j++){
        int start = 0;
        for(int i=0; i<L; i++){
            int a = z;
            for(int k=block_len[i][j].start; k < block_len[i][j].end; k++)
                organized_input[z++] = out[k];
            block_len[i][j].start = start;
            block_len[i][j].end = start + z - a;
            start = start + z - a;
        }
    }
    //return 0;

    int **d_ins;
    int **d_outs;
    d_ins = (int**) malloc(sizeof(int*) * S);
    d_outs = (int**) malloc(sizeof(int*) * S);
    int offset = 0;
    for(int i =0; i<S; i++){
        cudaMalloc(&d_ins[i],sizeof(int)*blocks_len[i]);
        cudaMemcpy(d_ins[i],organized_input + offset,sizeof(int) * blocks_len[i],cudaMemcpyHostToDevice);
        cudaMalloc(&d_outs[i],sizeof(int)*blocks_len[i]);
        offset += blocks_len[i];
    }

//    organized_input = organized_input + blocks_len[0];
//    cudaMemcpy(d_in, organized_input, size, cudaMemcpyHostToDevice);
    for(int s = 0; s < S; s++){
        for(int k=1; k <= (L/2); k*=2){
            for(int i = 0; i< L-k; i+=(k*2)){
                arb_merge<<<1,32>>>(d_ins[s],d_outs[s],block_len[i][s].start,block_len[i][s].len, block_len[i][s].start, block_len[i+k][s].len,block_len[i+k][s].start);
                cudaDeviceSynchronize();
                cudaMemcpy(out, d_outs[1], blocks_len[1]*sizeof(int), cudaMemcpyDeviceToHost);
                block_len[i][s].end = block_len[i+k][s].end;
                block_len[i][s].len = block_len[i][s].len + block_len[i+k][s].len;
                memset(&block_len[i+k][s],0,sizeof(block_info));
            }
            cudaMemcpy(d_ins[s],d_outs[s],blocks_len[s]*sizeof(int),cudaMemcpyDeviceToDevice);
            cudaMemcpy(organized_input,d_outs[s],blocks_len[s]*sizeof(int),cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize(); 
    
  //  cudaMemcpy(out, d_outs[1], blocks_len[1]*sizeof(int), cudaMemcpyDeviceToHost);
    offset = 0;
    for(int i =0; i<S; i++){
        cudaMemcpy(out + offset,d_outs[i],sizeof(int) * blocks_len[i],cudaMemcpyDeviceToHost);
        offset += blocks_len[i];
    }
    for(int i=0; i<n;i++)
        printf("%d ",out[i]);

    return 0;
}
