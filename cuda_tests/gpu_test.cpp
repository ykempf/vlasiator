#include "gpu_velocity_grid.hpp"

using namespace spatial_cell;

const unsigned int min_lim = 5;
const unsigned int max_lim = 10;

void print_elapsed_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n", milliseconds);
}

int main(void) {
    putchar('\n');
    init_spatial_cell_static();
    SpatialCell *spacell = create_maxwellian(1.0e6, 1.0e5);
    cudaEvent_t start, stop;
    // Initialize cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int ids_len = (int)2e4;
    int ids[ids_len];

    // Add blocks to the given ids
    /*
    for (int i=0; i<ids_len; i++) {
        int ind = i;
        ind3d inds = GPU_velocity_grid::get_velocity_block_indices_host(ind);
        // Skip if not inside the wanted area.
        if (inds.x < min_lim || inds.x > max_lim ||
            inds.y < min_lim || inds.y > max_lim ||
            inds.z < min_lim || inds.z > max_lim) continue;
        ids[i] = ind;
        spacell.add_velocity_block(ind);
        Velocity_Block* block_ptr = spacell.at(ind);
        // Put some data into each velocity spacell
        for (int j = 0; j < WID3; j++) block_ptr->data[j]=ind+j/100.;
    }
    */

    // Print data as it is on CPU
    //printf("On host:\n");
    //print_blocks(&spacell);
    //printf("Number of blocks: %i\n", ids_len);
    
    // Create a new instance. Constructor copies related data.
    printf("Create an instance of GPU_velovity_grid and copy data over to GPU:\n");
    CUDACALL(cudaDeviceSynchronize());
    cudaEventRecord(start);
    GPU_velocity_grid *ggrid = new GPU_velocity_grid(spacell);
    CUDACALL(cudaEventRecord(stop));
    cudaEventSynchronize(stop);
    print_elapsed_time(start, stop);
    
    /*
    // Print data from GPU
    printf("On GPU:\n");
    ggrid->print_blocks();
    printf("Print from kernel:\n");
    ggrid->k_print_blocks();
    */
    //print_constants();
    
    cudaEventRecord(start);
    unsigned int min_ind = ggrid->min_ind();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    putchar('\n');
    cudaEventRecord(start);
    min_ind = ggrid->min_ind();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("Min ind: %u\n", min_ind);
    ind3d min_indices = GPU_velocity_grid::get_velocity_block_indices_host(min_ind);
    printf("Min ind: %u (%u %u %u)\n", min_ind, min_indices.x, min_indices.y, min_indices.z);
    print_elapsed_time(start, stop);
    
    putchar('\n');
    cudaEventRecord(start);
    unsigned int max_ind = ggrid->max_ind();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("Max ind: %u\n", max_ind);
    ind3d max_indices = GPU_velocity_grid::get_velocity_block_indices_host(max_ind);
    printf("Max ind: %u (%u %u %u)\n", max_ind, max_indices.x, max_indices.y, max_indices.z);
    print_elapsed_time(start, stop);
    
    putchar('\n');
    printf("Grid initialization:\n");
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    ggrid->init_grid();
    CUDACALL(cudaEventRecord(stop));
    cudaEventSynchronize(stop);
    print_elapsed_time(start, stop);
    
    /*
    ggrid->print_cells();
    CUDACALL(cudaDeviceSynchronize());
    */
    /*
    putchar('\n');
    printf("Back to CPU:\n");
    CUDACALL(cudaDeviceSynchronize());
    cudaEventRecord(start);
    SpatialCell *spacell2 = ggrid->toSpatialCell();
    CUDACALL(cudaEventRecord(stop));
    cudaEventSynchronize(stop);
    print_elapsed_time(start, stop);
    */
    putchar('\n');
    return 0;
}
