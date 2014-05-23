#include "gpu_velocity_grid.hpp"

// TODO finish
// Same as SpatialCell::get_velocity_block_indices but revised for GPU. 
static vel_block_indices_t get_velocity_block_indices(const unsigned int blockid) {
    vel_block_indices_t indices;
//    indices = new vel_block_indices(3);
    indices[0] = blockid % SpatialCell::vx_length;
    indices[1] = (blockid / SpatialCell::vx_length) % SpatialCell::vy_length;
    indices[2] = blockid / (SpatialCell::vx_length * SpatialCell::vy_length);

    return indices;
}

void GPU_velocity_grid::print_blocks(void) {
    printf("Number of blocks: %4u.\n", *num_blocks);
    
    for (int i=0; i<*num_blocks; i++) {
        printf("%5.0u:\n", gpu_velocity_block_list[i]);
    }
}

