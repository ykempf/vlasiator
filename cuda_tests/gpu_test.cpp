#include "gpu_velocity_grid.hpp"

using namespace spatial_cell;

void print_elapsed_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n", milliseconds);
}

int main(void) {
    init_spatial_cell_static();
    SpatialCell cell;
    cudaEvent_t start, stop;    
    const int ids_len = (int)1e6;
    int ids[ids_len];

    // Add blocks to the given ids
    for (int i=0; i<ids_len; i++) {
        int ind = ids_len - i - 1;
        ids[i] = ind;
        cell.add_velocity_block(ind);
        Velocity_Block* block_ptr = cell.at(ind);
        block_ptr->data[0]=ind; // Put some data into each velocity cell
    }
    // Print data as it is on CPU
    //printf("On host:\n");
    //print_blocks(&cell);
    printf("%i %i %i %i %i %i\n", ids[0], ids[1], ids[2], ids[ids_len-3], ids[ids_len-2], ids[ids_len-1]);
    printf("Number of blocks: %i\n", ids_len);
    
    // Create a new instance. Constructor copies related data.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    GPU_velocity_grid *ggrid = new GPU_velocity_grid(&cell);
    
    /*
    // Print data from GPU
    printf("On GPU:\n");
    ggrid->print_blocks();
    printf("Print from kernel:\n");
    ggrid->k_print_blocks();
    */
    
    //print_constants();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    unsigned int min_ind = ggrid->min_ind(ids_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    print_elapsed_time(start, stop);
    printf("Min ind: %u\n", min_ind);
    ind3d min_indices = GPU_velocity_grid::get_velocity_block_indices_host(min_ind);
    printf("Min ind: %u (%u %u %u)\n", min_ind, min_indices.x, min_indices.y, min_indices.z);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    unsigned int max_ind = ggrid->max_ind(ids_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    print_elapsed_time(start, stop);
    printf("Max ind: %u\n", max_ind);
    ind3d max_indices = GPU_velocity_grid::get_velocity_block_indices_host(max_ind);
    printf("Max ind: %u (%u %u %u)\n", max_ind, max_indices.x, max_indices.y, max_indices.z);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    min_ind = ggrid->min_ind(ids_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    print_elapsed_time(start, stop);
    printf("Min ind: %u\n", min_ind);
    min_indices = GPU_velocity_grid::get_velocity_block_indices_host(min_ind);
    printf("Min ind: %u (%u %u %u)\n", min_ind, min_indices.x, min_indices.y, min_indices.z);
    
    return 0;
}
