/*  
 *  Simple reduction
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include "culock.hpp"

#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


const int N = 128;
const float val = 0.5f;

__global__ void sum(float *dSum, float *dData, Lock lock){
    extern __shared__ float temp[];
    int tid=threadIdx.x;
    temp[tid]=dData[tid+blockIdx.x*blockDim.x]; // Read data in
    int d;
    for (d=blockDim.x>>1;d>=1;d>>=1){
        __syncthreads();
        if (tid<d) temp[tid]+=temp[tid+d];
    }
    if (tid==0){
        lock.lock();
        *dSum+=temp[0];
        __threadfence(); // Make sure the memory write is complete before letting other threads access it
        lock.unlock();
    }
}

__global__ void min_reduce(float *dSum, float *dData, Lock lock){
    extern __shared__ float temp[];
    int tid=threadIdx.x;
    temp[tid]=dData[tid+blockIdx.x*blockDim.x]; // Read data in
    int d;
    for (d=blockDim.x>>1;d>=1;d>>=1){
        __syncthreads();
        if (tid<d) temp[tid] = fmin(temp[tid], temp[tid+d]);
    }
    if (tid==0){
        lock.lock();
        *dSum=fmin(*dSum, temp[0]); // Make sure the memory write is complete before letting other threads access it
        __threadfence();
        lock.unlock();
    }
}

void sum_test() {
    const float zero = 0.0f;
    
    Lock lock;
    
    size_t size = N * sizeof(float);
    
    float *d_arr;
    float *h_arr;
    float *result;
    
    h_arr = (float*)malloc(size);
    CUDACALL(cudaMallocManaged(&d_arr, size));
    CUDACALL(cudaMallocManaged(&result, sizeof(float)));
    
    for(int i=0; i<N; i++) h_arr[i] = val;
    memcpy(d_arr, h_arr, size);
    memcpy(result, &zero, sizeof(float));
    sum<<<4,32,4000>>>(result, d_arr, lock);
    cudaDeviceSynchronize();
    printf("Reduced value: %f\n", *result);
    printf("Expected: %f\n", N*val);
}

void min_test() {
    const float max_float = std::numeric_limits<float>::max();
    
    Lock lock;
    
    size_t size = N * sizeof(float);
    
    float *d_arr;
    float *h_arr;
    float *result;
    
    h_arr = (float*)malloc(size);
    CUDACALL(cudaMallocManaged(&d_arr, size));
    CUDACALL(cudaMallocManaged(&result, sizeof(float)));
    
    for(int i=0; i<N; i++) h_arr[i] = val;
    memcpy(d_arr, h_arr, size);
    memcpy(result, &max_float, sizeof(float));
    min_reduce<<<4,32,2000>>>(result, d_arr, lock);
    cudaDeviceSynchronize();
    printf("Reduced value: %f\n", *result);
    printf("Expected: %f\n", val);
}

int main(void) {
    printf("Summation:\n");
    sum_test();
    printf("\nMinimum value:\n");
    min_test();

    return 0;
}
