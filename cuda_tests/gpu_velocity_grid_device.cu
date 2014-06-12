/*
 *  Anything that needs to be translated by nvcc goes here.
 */

#include "gpu_velocity_grid.hpp"
#include "culock.hpp"

extern int vx_length, vy_length, vz_length;

__global__ void kernel_print_blocks(GPU_velocity_grid grid) {
    unsigned int tid = blockIdx.x;
    unsigned int ind;
    ind3d indices;
    ind = grid.velocity_block_list[tid];
    indices = grid.get_velocity_block_indices(ind);
    printf("%5.0u: (%4i, %4i, %4i) %7.1f\n", ind, indices.ind[0], indices.ind[1], indices.ind[2], grid.block_data[tid*WID3]);
}

// Same as SpatialCell::get_velocity_block_indices but revised for GPU. This must stay in this file so that both host and device versions can be created.
__device__ ind3d GPU_velocity_grid::get_velocity_block_indices(const unsigned int blockid) {
    ind3d indices;
    indices.ind[0] = blockid % vx_len();
    indices.ind[1] = (blockid / vx_len()) % vy_len();
    indices.ind[2] = blockid / (vx_len() * vy_len());

    return indices;
}

// Wrapper for the kernel
__host__ void GPU_velocity_grid::k_print_blocks(void) {
    kernel_print_blocks<<<*num_blocks, 1>>>(*this);
    CUDACALL(cudaPeekAtLastError()); // Check for kernel launch errors
    CUDACALL(cudaDeviceSynchronize()); // Check for other cuda errors
}

