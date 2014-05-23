#include "gpu_velocity_grid.hpp"

// Copies velocity_block_list and block_data as well as necessary constants from a SpatialCell to GPU for processing.
GPU_velocity_grid::GPU_velocity_grid(SpatialCell *spacell) {
	
	
    // Allocate memory on the gpu
	unsigned int vel_block_list_size = spacell->number_of_blocks*sizeof(unsigned int);
	unsigned int block_data_size = spacell->block_data.size()*sizeof(float);
    cudaMallocManaged(&num_blocks, sizeof(unsigned int));
    cudaMallocManaged(&vx_length, sizeof(unsigned int));
    cudaMallocManaged(&vy_length, sizeof(unsigned int));
    cudaMallocManaged(&vz_length, sizeof(unsigned int));
	cudaMallocManaged(&velocity_block_list, vel_block_list_size);
	cudaMallocManaged(&block_data, block_data_size);
	
	// Copy to gpu
	unsigned int *velocity_block_list_arr = &(spacell->velocity_block_list[0]);
	float *block_data_arr = &(spacell->block_data[0]);
	memcpy(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int));
    memcpy(vx_length, &(SpatialCell::vx_length), sizeof(unsigned int));
    memcpy(vy_length, &(SpatialCell::vy_length), sizeof(unsigned int));
    memcpy(vz_length, &(SpatialCell::vz_length), sizeof(unsigned int));
	memcpy(velocity_block_list, velocity_block_list_arr, vel_block_list_size);
	memcpy(block_data, block_data_arr, block_data_size);
}
