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

// Same as SpatialCell::get_velocity_block_indices but revised for GPU. 
vel_block_indices_t GPU_velocity_grid::get_velocity_block_indices(const unsigned int blockid) {
    vel_block_indices_t indices;
    indices.ind[0] = blockid % *vx_length;
    indices.ind[1] = (blockid / *(vx_length)) % *vy_length;
    indices.ind[2] = blockid / (*vx_length * *vy_length);

    return indices;
}

// Prints the same data as print_blocks from gpu_test.cpp
void GPU_velocity_grid::print_blocks(void) {
    printf("Number of blocks: %4u.\n", *num_blocks);
    unsigned int ind;
    vel_block_indices_t indices;
    for (int i=0; i<*num_blocks; i++) {
        ind = velocity_block_list[i];
        printf("%5.0u: ", ind);
        indices = get_velocity_block_indices(ind);
        printf("(%4i, %4i, %4i) %7.1f\n", indices.ind[0], indices.ind[1], indices.ind[2], block_data[i*WID3]);
    }
}
