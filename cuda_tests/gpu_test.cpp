#include "gpu_velocity_grid.hpp"


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
