#include "gpu_velocity_grid.hpp"

using namespace spatial_cell;

// Constant memory can not be allocated inside class definition
// Identical to those of SpatialCell aka. dimensions of velocity space.
__constant__ int vx_length, \
                 vy_length, \
                 vz_length;

// Copies velocity_block_list and block_data as well as necessary constants from a SpatialCell to GPU for processing.
GPU_velocity_grid::GPU_velocity_grid(SpatialCell *spacell) {
	
	extern int vx_length, vy_length, vz_length;
	
    // Allocate memory on the gpu
	unsigned int vel_block_list_size = spacell->number_of_blocks*sizeof(unsigned int);
	unsigned int block_data_size = spacell->block_data.size()*sizeof(float);
    cudaMallocManaged(&num_blocks, sizeof(unsigned int));
	cudaMallocManaged(&velocity_block_list, vel_block_list_size);
	cudaMallocManaged(&block_data, block_data_size);
	
	// Copy to gpu
	unsigned int *velocity_block_list_arr = &(spacell->velocity_block_list[0]);
	float *block_data_arr = &(spacell->block_data[0]);
	memcpy(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int));
	cudaMemcpyToSymbol(vx_length, &SpatialCell::vx_length, sizeof(int));
	cudaMemcpyToSymbol(vy_length, &SpatialCell::vy_length, sizeof(int));
	cudaMemcpyToSymbol(vz_length, &SpatialCell::vz_length, sizeof(int));
    //memcpy(vx_length, &(SpatialCell::vx_length), sizeof(unsigned int));
    //memcpy(vy_length, &(SpatialCell::vy_length), sizeof(unsigned int));
    //memcpy(vz_length, &(SpatialCell::vz_length), sizeof(unsigned int));
	memcpy(velocity_block_list, velocity_block_list_arr, vel_block_list_size);
	memcpy(block_data, block_data_arr, block_data_size);
}

GPU_velocity_grid::~GPU_velocity_grid() {
    // Free memory
    cudaFree(num_blocks);
	cudaFree(velocity_block_list);
	cudaFree(block_data);
}

// Simple accessors
__device__ int GPU_velocity_grid::vx_len(void) {return vx_length;}
__device__ int GPU_velocity_grid::vy_len(void) {return vy_length;}
__device__ int GPU_velocity_grid::vz_len(void) {return vz_length;}

// Same as SpatialCell::get_velocity_block_indices but revised for GPU
__device__ ind3d GPU_velocity_grid::get_velocity_block_indices(const unsigned int blockid) {
    ind3d indices;
    indices.ind[0] = blockid % vx_length;
    indices.ind[1] = (blockid / vx_length) % vy_length;
    indices.ind[2] = blockid / (vx_length * vy_length);

    return indices;
}
// Host version. Requires initialized SpatialCell static variables.
__host__ ind3d GPU_velocity_grid::get_velocity_block_indices_host(const unsigned int blockid) {
    ind3d indices;
    indices.ind[0] = blockid % SpatialCell::vx_length;
    indices.ind[1] = (blockid / SpatialCell::vx_length) % SpatialCell::vy_length;
    indices.ind[2] = blockid / (SpatialCell::vx_length * SpatialCell::vy_length);

    return indices;
}

// Same as print_blocks, but prints from a kernel
__global__ void kernel_print_blocks(GPU_velocity_grid grid) {
    unsigned int tid = blockIdx.x;
    unsigned int ind;
    ind3d indices;
    ind = grid.velocity_block_list[tid];
    indices = GPU_velocity_grid::get_velocity_block_indices(ind);
    printf("%5.0u: (%4i, %4i, %4i) %7.1f\n", ind, indices.ind[0], indices.ind[1], indices.ind[2], grid.block_data[tid*WID3]);
}

// Wrapper for the kernel
__host__ void GPU_velocity_grid::k_print_blocks(void) {
    kernel_print_blocks<<<*num_blocks, 1>>>(*this);
    CUDACALL(cudaPeekAtLastError()); // Check for kernel launch errors
    CUDACALL(cudaDeviceSynchronize()); // Check for other cuda errors
}

void GPU_velocity_grid::print_blocks(void) {
    printf("Number of blocks: %4u.\n", *num_blocks);
    unsigned int ind;
    ind3d indices;
    for (int i=0; i<*num_blocks; i++) {
        ind = velocity_block_list[i];
        printf("%5.0u: ", ind);
        indices = get_velocity_block_indices_host(ind);
        printf("(%4i, %4i, %4i) %7.1f\n", indices.ind[0], indices.ind[1], indices.ind[2], block_data[i*WID3]);
    }
}
