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

int main(void) {
    init_spatial_cell();
    SpatialCell cell;
    
    int ids_len = 3;
    int ids[] = {1, 5, 10};
    
    for (int i=0; i<ids_len; i++) {
        cell.add_velocity_block(i);
    }
    
    return 0;
}
