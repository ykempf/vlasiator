#include "gpu_velocity_grid.hpp"

// Use same values for all dimensions for this test
const int spatial_cell_side_length = 100;
const float v_min = 1e-5;
const float v_max = 1e10;

using namespace spatial_cell;

// Initializes the SpatialCell static variables to values given above
void init_spatial_cell(void) {
    spatial_cell::SpatialCell::vx_length = spatial_cell_side_length;
    spatial_cell::SpatialCell::vy_length = spatial_cell_side_length;
    spatial_cell::SpatialCell::vz_length = spatial_cell_side_length;
    spatial_cell::SpatialCell::vx_min = v_min;
    spatial_cell::SpatialCell::vx_max = v_max;
    spatial_cell::SpatialCell::vy_min = v_min;
    spatial_cell::SpatialCell::vy_max = v_max;
    spatial_cell::SpatialCell::vz_min = v_min;
    spatial_cell::SpatialCell::vz_max = v_max;
    //spatial_cell::SpatialCell::velocity_block_min_value = P::sparseMinValue; // Not currently used
    
    
    // Rest are calculated from the previous
    spatial_cell::SpatialCell::max_velocity_blocks =
    spatial_cell::SpatialCell::vx_length * spatial_cell::SpatialCell::vy_length * \
                                                spatial_cell::SpatialCell::vz_length;
    spatial_cell::SpatialCell::grid_dvx = spatial_cell::SpatialCell::vx_max - \
                                                spatial_cell::SpatialCell::vx_min;
    spatial_cell::SpatialCell::grid_dvy = spatial_cell::SpatialCell::vy_max - \
                                                spatial_cell::SpatialCell::vy_min;
    spatial_cell::SpatialCell::grid_dvz = spatial_cell::SpatialCell::vz_max - \
                                                spatial_cell::SpatialCell::vz_min;
    spatial_cell::SpatialCell::block_dvx = spatial_cell::SpatialCell::grid_dvx / \
                                                spatial_cell::SpatialCell::vx_length;
    spatial_cell::SpatialCell::block_dvy = spatial_cell::SpatialCell::grid_dvy / \
                                                spatial_cell::SpatialCell::vy_length;
    spatial_cell::SpatialCell::block_dvz = spatial_cell::SpatialCell::grid_dvz / \
                                                spatial_cell::SpatialCell::vz_length;
    spatial_cell::SpatialCell::cell_dvx = spatial_cell::SpatialCell::block_dvx / \
                                                block_vx_length;
    spatial_cell::SpatialCell::cell_dvy = spatial_cell::SpatialCell::block_dvy / block_vy_length;
    spatial_cell::SpatialCell::cell_dvz = spatial_cell::SpatialCell::block_dvz / block_vz_length;
}

// Prints information about the velocity blocks on the CPU. Used to check correct transfer to GPU.
void print_blocks(SpatialCell *cell) {
    spatial_cell::velocity_block_indices_t indices;
    unsigned int ind;
    for(unsigned int i=0; i<cell->number_of_blocks; i++) {
        ind = cell->velocity_block_list[i];
        printf("%5.0u: ", ind);
        Velocity_Block* block_ptr = cell->at(ind);
        indices = SpatialCell::get_velocity_block_indices(ind);
        printf("(%4i, %4i, %4i) %7.1f\n", indices[0], indices[1], indices[2], block_ptr->data[0]);
    }
}


int main(void) {
    init_spatial_cell();
    SpatialCell cell;
    
    const int ids_len = 8;
    int ids[] = {1, 10, 100, 101, 999, 1001, 9999, 10005};

    // Add blocks to the given ids
    for (int i=0; i<ids_len; i++) {
        int ind = ids[i];
        cell.add_velocity_block(ind);
        Velocity_Block* block_ptr = cell.at(ind);
        block_ptr->data[0]=ind; // Put some data into each velocity cell
    }
    // Print data as it is on CPU
    printf("On host:\n");
    print_blocks(&cell);
    
    // Create a new instance. Constructor copies related data.
    GPU_velocity_grid *ggrid = new GPU_velocity_grid(&cell);
    
    // Print data from GPU
    printf("On GPU:\n");
    ggrid->print_blocks();
    printf("Print from kernel:\n");
    ggrid->k_print_blocks();
    return 0;
}
