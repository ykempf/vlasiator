#include "gpu_velocity_grid.hpp"

int GPU_velocity_grid::transfer2gpu(SpatialCell *spacell) {
	/* To be transferred to GPU (from a SpatialCell)
	unsigned int spacell->number_of_blocks
	std::vector<unsigned int> velocity_block_list
	std::vector<Realf,aligned_allocator<Realf,64> > block_data
	*/
	
	unsigned int *gpu_number_of_blocks;
	cudaMallocManaged(&gpu_number_of_blocks, 1*sizeof(unsigned int));
	*gpu_number_of_blocks = spacell->number_of_blocks;
	
	// Allocate memory on the gpu
	unsigned int vel_block_list_size = spacell->number_of_blocks*sizeof(unsigned int);
	unsigned int *gpu_velocity_block_list;
	cudaMallocManaged(&gpu_velocity_block_list, vel_block_list_size); // Using unified memory
	
	// Copy to gpu
	unsigned int *velocity_block_list_arr = spacell->velocity_block_list[0];
	memcpy(gpu_velocity_block_list, velocity_block_list_arr, vel_block_list_size);
}
