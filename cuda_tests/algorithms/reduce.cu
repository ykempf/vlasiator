/*  
 *  Simple reduction
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include "culock.hpp"

// A wrapper for cuda function calls (made by host) to check for and report errors returned by the function
#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


const int N = 64*64*3;
const float val = 0.5f;
const unsigned int block_size = 64;

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

__global__ void sum2(float *data, unsigned int data_size, float *resultarr) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < data_size) { // Make sure to stay in bounds
        sdata[tid] = data[i];
    }
    else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    //printf("%4i %3.1f %3.1f\n", i, sdata[tid], sdata[0]);
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) {
        if (blockIdx.x == 0) printf("len:%i \n", data_size);
        printf("%3i: %f\n", blockIdx.x, sdata[0]);
        resultarr[blockIdx.x] = sdata[0];
    }
}

// Returns the ceiling of the equivalent float division
template<typename T>
inline T ceilDivide(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// Sums the array data of size len on GPU and returns the sum.
float sum_reduce(float *data, unsigned int len) {
    unsigned int grid_size = ceilDivide(len, block_size);
    float *result;
    unsigned int last_grid_size = grid_size;
    
    CUDACALL(cudaMallocManaged(&result, grid_size*sizeof(float))); // Only 1 element is used by each block
    
    printf("Grid:%u, input size:%u\n", grid_size, len);
    // First round works on the input array
    printf("First invocation!!!!!!!!!!!!!!!!!\n");
    sum2<<<grid_size, block_size, block_size*sizeof(float)>>>(data, len, result);
    cudaDeviceSynchronize();
    // Now iterate over the result array
    grid_size = ceilDivide(grid_size, block_size); // This is essentially ceiling(grid_size / block_size)
    len = ceilDivide(len, block_size);
    //len /= block_size;
    printf("Entering loop!!!!!!!!!!!!!!!!!!!!\n");
    for (; grid_size > 1; grid_size = ceilDivide(grid_size, block_size)) {
        sum2<<<grid_size, block_size, block_size*sizeof(float)>>>(result, len, result);
        cudaDeviceSynchronize();
        len = ceilDivide(len, block_size);
        last_grid_size = grid_size;
    }
    // Final iteration
    printf("Last kernel call!!!!!!!!!!!!!!!!!!1     %u\n", last_grid_size);
    for (int i=0; i<last_grid_size; i++) printf("%4.1f ", result[i]);
    putchar('\n');
    sum2<<<1, block_size, block_size*sizeof(float)>>>(result, last_grid_size, result);

    //for (int i=0; i<last_grid_size; i++) printf("%4.1f ", result[i]);
    //putchar('\n');
    
    // Transfer the result to host
    float h_result;
    cudaMemcpy(&h_result, result, sizeof(float), cudaMemcpyDeviceToHost);

    return h_result;
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

void sum2_test() {
    size_t size = N * sizeof(float);
    
    float *d_arr;
    float *h_arr;
    
    // Allocate
    h_arr = (float*)malloc(size);
    CUDACALL(cudaMallocManaged(&d_arr, size));
    
    // Initialize and copy
    for(int i=0; i<N; i++) h_arr[i] = val;
    memcpy(d_arr, h_arr, size);

    // Run kernel and get result
    float result = sum_reduce(d_arr, N);
    printf("Reduced value: %f\n", result);
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
    printf("\nSummation 2:\n");
    sum2_test();
    //printf("\nMinimum value:\n");
    //min_test();

    return 0;
}
