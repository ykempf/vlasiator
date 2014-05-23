#include "gpu_velocity_grid.hpp"

// Same as SpatialCell::get_velocity_block_indices but revised for GPU. 
vel_block_indices_t GPU_velocity_grid::get_velocity_block_indices(const unsigned int blockid) {
    vel_block_indices_t indices(3);
    indices[0] = blockid % *vx_length;
    indices[1] = (blockid / *(vx_length)) % *vy_length;
    indices[2] = blockid / (*vx_length * *vy_length);

    return indices;
}

// Prints the same data as print_blocks from gpu_test.cpp
void GPU_velocity_grid::print_blocks(void) {
    printf("Number of blocks: %4u.\n", *num_blocks);
    unsigned int ind;
    vel_block_indices_t indices(3);
    for (int i=0; i<*num_blocks; i++) {
        ind = velocity_block_list[i];
        printf("%5.0u: ", ind);
        indices = get_velocity_block_indices(ind);
        printf("(%4i, %4i, %4i) %7.1f\n", indices[0], indices[1], indices[2], block_data[i*WID3]);
    }
}

