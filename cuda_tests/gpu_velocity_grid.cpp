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
    cudaMallocManaged(&vx_length, sizeof(unsigned int));
    cudaMallocManaged(&vy_length, sizeof(unsigned int));
    cudaMallocManaged(&vz_length, sizeof(unsigned int));
	cudaMallocManaged(&velocity_block_list, vel_block_list_size);
	
	// Copy to gpu
	unsigned int *velocity_block_list_arr = &(spacell->velocity_block_list[0]);
	//cudaMemcpyToSymbol(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int));
    memcpy(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int));
    memcpy(vx_length, &(SpatialCell::vx_length), sizeof(unsigned int));
    memcpy(vy_length, &(SpatialCell::vy_length), sizeof(unsigned int));
    memcpy(vz_length, &(SpatialCell::vz_length), sizeof(unsigned int));
	memcpy(velocity_block_list, velocity_block_list_arr, vel_block_list_size);
}
