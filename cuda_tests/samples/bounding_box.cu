/*  
 *  Simple reduction
 */ 

#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int lock = 0;

__global__ void sum(float *dSum, float *dData){
    extern __shared__ float temp[];
    int tid=threadIdx.x;
    temp[tid]=dData[tid+blockIdx.x*blockDim.x]; // Read data in
    int d;
    for (d=blockDim.x>>1;d>=1;d>>=1){
        __syncthreads();
        if (tid<d) temp[tid]+=temp[tid+d];
    }
    if (tid==0){
        while (atomicCas(&lock,0,1)){} // Wait for lock to open
        *dSum+=temp[0];
        __threadfence();
        lock=0;
    }
}

int main(void) {
    const int N = 100;
    
    size_t size = N * sizeof(float);
    
    float *d_arr;
    float *h_arr;
    float *result;
    
    h_arr = (float*)malloc(size);
    cudaMallocManaged(&d_arr, size);
    cudaMallocManaged(&result, 1*sizeof(float));
    
    for(int i=0; i<N; i++) h_arr[i] = 0.1f;
    memcpy(d_arr, h_arr, size);
    sum<<<50,2>>>(result, d_arr);
    printf("%f", result);
}
