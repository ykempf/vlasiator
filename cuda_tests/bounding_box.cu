#include "gpu_velocity_grid.hpp"

#define REDUCE_BLOCK_SIZE 64u


/*
 *  MINIMuM INDEX
 */

__device__ unsigned int lower_bound(const unsigned int point1, const unsigned int point2) {
    ind3d indices1 = GPU_velocity_grid::get_velocity_block_indices(point1);
    ind3d indices2 = GPU_velocity_grid::get_velocity_block_indices(point2);
    // Put the smaller value of each index to indices1
    indices1.x = indices1.x < indices2.x ? indices1.x : indices2.x;
    indices1.y = indices1.y < indices2.y ? indices1.y : indices2.y;
    indices1.z = indices1.z < indices2.z ? indices1.z : indices2.z;
    // Return as 1d index
    return GPU_velocity_grid::get_velocity_block(indices1);
}

// Finds the minimum index in all dimensions and returns it as a 1d index
__global__ void min_ind_kernel(unsigned int *data, unsigned int data_size, unsigned int *resultarr) {
    extern __shared__ unsigned int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < data_size) { // Make sure to stay in bounds
        sdata[tid] = data[i];
    }
    else { // Otherwise initialize with last value from the list
        sdata[tid] = data[data_size-1];
    }
    __syncthreads();

    //printf("%4i %3.1f %3.1f\n", i, sdata[tid], sdata[0]);
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = lower_bound(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) {
        #ifdef DEBUG_PRINT
        if (blockIdx.x == 0) printf("len:%i \n", data_size);
        printf("%3i: %f\n", blockIdx.x, sdata[0]);
        #endif
        resultarr[blockIdx.x] = sdata[0];
    }
}

// Finds the minimum 1d index of the velocity blocks in this spatial cell
__host__ unsigned int GPU_velocity_grid::min_ind(void) {
    unsigned int len;
    unsigned int grid_size;
    unsigned int *result;
    unsigned int last_grid_size;
    CUDACALL(cudaMemcpy(&len, num_blocks, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    grid_size = ceilDivide(len, REDUCE_BLOCK_SIZE);
    last_grid_size = grid_size;
    #ifdef CEBUG_PRINT
    printf("len: %u\n", len);
    #endif
    CUDACALL(cudaMalloc(&result, grid_size*sizeof(unsigned int))); // Only 1 element is used by each block
    #ifdef DEBUG_PRINT
    printf("Grid:%u, input size:%u\n", grid_size, len);
    // First round works on the input array
    printf("First invocation!!!!!!!!!!!!!!!!!\n");
    #endif
    min_ind_kernel<<<grid_size, REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE*sizeof(unsigned int)>>>(velocity_block_list, len, result);
    CUDACALL(cudaDeviceSynchronize());
    // Now iterate over the result array
    grid_size = ceilDivide(grid_size, REDUCE_BLOCK_SIZE);
    len = ceilDivide(len, REDUCE_BLOCK_SIZE);
    #ifdef DEBUG_PRINT
    printf("Entering loop!!!!!!!!!!!!!!!!!!!!\n");
    #endif
    for (; grid_size > 1; grid_size = ceilDivide(grid_size, REDUCE_BLOCK_SIZE)) {
        min_ind_kernel<<<grid_size, REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE*sizeof(unsigned int)>>>(result, len, result);
        CUDACALL(cudaDeviceSynchronize());
        len = ceilDivide(len, REDUCE_BLOCK_SIZE);
        last_grid_size = grid_size;
    }
    // Final iteration
    #ifdef DEBUG_PRINT
    printf("Last kernel call!!!!!!!!!!!!!!!!!!1     %u\n", last_grid_size);
    #endif
    min_ind_kernel<<<1, REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE*sizeof(unsigned int)>>>(result, last_grid_size, result);

    //for (int i=0; i<last_grid_size; i++) printf("%4.1f ", result[i]);
    //putchar('\n');

    // Transfer the result to host
    unsigned int h_result;
    CUDACALL(cudaMemcpy(&h_result, result, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDACALL(cudaFree(result));
    return h_result;
}

/*
 *  MAXIMUM INDEX
 */

__device__ unsigned int upper_bound(const unsigned int point1, const unsigned int point2) {
    ind3d indices1 = GPU_velocity_grid::get_velocity_block_indices(point1);
    ind3d indices2 = GPU_velocity_grid::get_velocity_block_indices(point2);
    // Put the smaller value of each index to indices1
    indices1.x = indices1.x > indices2.x ? indices1.x : indices2.x;
    indices1.y = indices1.y > indices2.y ? indices1.y : indices2.y;
    indices1.z = indices1.z > indices2.z ? indices1.z : indices2.z;
    // Return as 1d index
    return GPU_velocity_grid::get_velocity_block(indices1);
}

// Finds the maximum index in all dimensions and returns it as a 1d index
__global__ void max_ind_kernel(unsigned int *data, unsigned int data_size, unsigned int *resultarr) {
    extern __shared__ unsigned int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < data_size) { // Make sure to stay in bounds
        sdata[tid] = data[i];
    }
    else { // Otherwise initialize with smallest possible index
        sdata[tid] = 0;
    }
    __syncthreads();

    //printf("%4i %3.1f %3.1f\n", i, sdata[tid], sdata[0]);
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = upper_bound(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) {
        #ifdef DEBUG_PRINT
        if (blockIdx.x == 0) printf("len:%i \n", data_size);
        printf("%3i: %f\n", blockIdx.x, sdata[0]);
        #endif
        resultarr[blockIdx.x] = sdata[0];
    }
}

// Finds the maximum 1d index of the velocity blocks in this spatial cell
__host__ unsigned int GPU_velocity_grid::max_ind(void) {
    unsigned int len;
    unsigned int grid_size;
    unsigned int *result;
    unsigned int last_grid_size;
    CUDACALL(cudaMemcpy(&len, num_blocks, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    grid_size = ceilDivide(len, REDUCE_BLOCK_SIZE);
    last_grid_size = grid_size;
    #ifdef DEBUG_PRINT
    printf("len: %u\n", len);
    #endif
    CUDACALL(cudaMalloc(&result, grid_size*sizeof(unsigned int))); // Only 1 element is used by each block
    #ifdef DEBUG_PRINT
    printf("Grid:%u, input size:%u\n", grid_size, len);
    // First round works on the input array
    printf("First invocation!!!!!!!!!!!!!!!!!\n");
    #endif
    max_ind_kernel<<<grid_size, REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE*sizeof(unsigned int)>>>(velocity_block_list, len, result);
    CUDACALL(cudaDeviceSynchronize());
    // Now iterate over the result array
    grid_size = ceilDivide(grid_size, REDUCE_BLOCK_SIZE);
    len = ceilDivide(len, REDUCE_BLOCK_SIZE);
    #ifdef DEBUG_PRINT
    printf("Entering loop!!!!!!!!!!!!!!!!!!!!\n");
    #endif
    for (; grid_size > 1; grid_size = ceilDivide(grid_size, REDUCE_BLOCK_SIZE)) {
        max_ind_kernel<<<grid_size, REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE*sizeof(unsigned int)>>>(result, len, result);
        CUDACALL(cudaDeviceSynchronize());
        len = ceilDivide(len, REDUCE_BLOCK_SIZE);
        last_grid_size = grid_size;
    }
    // Final iteration
    #ifdef DEBUG_PRINT
    printf("Last kernel call!!!!!!!!!!!!!!!!!!1     %u\n", last_grid_size);
    #endif
    max_ind_kernel<<<1, REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE*sizeof(unsigned int)>>>(result, last_grid_size, result);

    //for (int i=0; i<last_grid_size; i++) printf("%4.1f ", result[i]);
    //putchar('\n');

    // Transfer the result to host
    unsigned int h_result;
    CUDACALL(cudaMemcpy(&h_result, result, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDACALL(cudaFree(result));
    return h_result;
}
