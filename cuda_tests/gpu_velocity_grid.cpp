#include "gpu_velocity_grid.hpp"

// Copies velocity_block_list and block_data from a SpatialCell to GPU for processing.
GPU_velocity_grid::GPU_velocity_grid(SpatialCell *spacell) {
	/* To be transferred to GPU (from a SpatialCell)
	unsigned int spacell->number_of_blocks
	std::vector<unsigned int> velocity_block_list
	std::vector<Realf,aligned_allocator<Realf,64> > block_data
	*/
	
	
    // Allocate memory on the gpu
	unsigned int vel_block_list_size = spacell->number_of_blocks*sizeof(unsigned int);
	cudaMallocManaged(&num_blocks, sizeof(unsigned int));
	cudaMallocManaged(&gpu_velocity_block_list, vel_block_list_size);
	
	// Copy to gpu
	unsigned int *velocity_block_list_arr = &(spacell->velocity_block_list[0]);
	memcpy(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int));
	memcpy(gpu_velocity_block_list, velocity_block_list_arr, vel_block_list_size);
}
