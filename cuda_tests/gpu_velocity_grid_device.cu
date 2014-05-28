/*
 *  Anything that needs to be translated by nvcc goes here.
 */

#include "gpu_velocity_grid.hpp"
#include "culock.hpp"

__global__ void kernel_print_blocks(GPU_velocity_grid grid) {
    unsigned int tid = blockIdx.x;
    unsigned int ind;
    vel_block_indices_t indices;
    ind = grid.velocity_block_list[tid];
    indices = grid.get_velocity_block_indices(ind);
    printf("%5.0u: (%4i, %4i, %4i) %7.1f\n", ind, indices.ind[0], indices.ind[1], indices.ind[2], grid.block_data[tid*WID3]);
}

// Same as SpatialCell::get_velocity_block_indices but revised for GPU. This must stay in this file so that both host and device versions can be created.
__host__ __device__ vel_block_indices_t GPU_velocity_grid::get_velocity_block_indices(const unsigned int blockid) {
    vel_block_indices_t indices;
    indices.ind[0] = blockid % *vx_length;
    indices.ind[1] = (blockid / *(vx_length)) % *vy_length;
    indices.ind[2] = blockid / (*vx_length * *vy_length);

    return indices;
}

// Wrapper for the kernel
__host__ void GPU_velocity_grid::k_print_blocks(void) {
    kernel_print_blocks<<<*num_blocks, 1>>>(*this);
    CUDACALL(cudaPeekAtLastError()); // Check for kernel launch errors
    CUDACALL(cudaDeviceSynchronize()); // Check for other cuda errors
}


// Finds the smallest indices in all three dimensions. Returned via (a separately allocated) result array.
__global__ void min_inds(GPU_velocity_grid grid, Lock lock, float *result){
    extern __shared__ float temp[];
    unsigned int *inds = grid.velocity_block_list;
    int tid=threadIdx.x;
    temp[tid]=inds[tid+blockIdx.x*blockDim.x]; // Read data in
    int d;
    for (d=blockDim.x>>1;d>=1;d>>=1){
        __syncthreads();
        if (tid<d) temp[tid] = fmin(temp[tid], temp[tid+d]);
    }
    if (tid==0){
        lock.lock();
        *result=fmin(*result, temp[0]); // Make sure the memory write is complete before letting other threads access it
        __threadfence();
        lock.unlock();
    }
}


