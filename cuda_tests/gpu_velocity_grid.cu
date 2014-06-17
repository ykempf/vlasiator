#include "gpu_velocity_grid.hpp"

using namespace spatial_cell;

// Constant memory can not be allocated inside class definition, therefore only accessed directly from this file and via accessors if necessary.
// Identical to those of SpatialCell aka. dimensions of velocity space.
__constant__ unsigned int vx_length, \
                          vy_length, \
                          vz_length;
__constant__ ind3d min3d, max3d;
__constant__ unsigned int min1d, max1d;

// Call with only 1 thread
__global__ void print_constants_k(void) {
    printf("vx_length: %u, vy_length: %u, vz_length: %u\n", vx_length, vy_length, vz_length);
}

void print_constants(void) {
    // Easiest to print from a kernel
    print_constants_k<<<1,1>>>();
}

// Copies velocity_block_list and block_data as well as necessary constants from a SpatialCell to GPU for processing.
GPU_velocity_grid::GPU_velocity_grid(SpatialCell *spacell) {
	
    // Allocate memory on the gpu
	unsigned int vel_block_list_size = spacell->number_of_blocks*sizeof(unsigned int);
	unsigned int block_data_size = spacell->block_data.size()*sizeof(float);
    cudaMallocManaged(&num_blocks, sizeof(unsigned int));
	cudaMallocManaged(&velocity_block_list, vel_block_list_size);
	cudaMallocManaged(&block_data, block_data_size);
	
	// Copy to gpu
	unsigned int *velocity_block_list_arr = &(spacell->velocity_block_list[0]);
	float *block_data_arr = &(spacell->block_data[0]);
	memcpy(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int));
	cudaMemcpyToSymbol(vx_length, &SpatialCell::vx_length, sizeof(unsigned int));
	cudaMemcpyToSymbol(vy_length, &SpatialCell::vy_length, sizeof(unsigned int));
	cudaMemcpyToSymbol(vz_length, &SpatialCell::vz_length, sizeof(unsigned int));
	memcpy(velocity_block_list, velocity_block_list_arr, vel_block_list_size);
	memcpy(block_data, block_data_arr, block_data_size);
}

GPU_velocity_grid::~GPU_velocity_grid() {
    // Free memory
    cudaFree(num_blocks);
	cudaFree(velocity_block_list);
	cudaFree(block_data);
	cudaFree(vel_grid);
}

// Simple accessors
__device__ inline unsigned int GPU_velocity_grid::vx_len(void) {return vx_length;}
__device__ inline unsigned int GPU_velocity_grid::vy_len(void) {return vy_length;}
__device__ inline unsigned int GPU_velocity_grid::vz_len(void) {return vz_length;}

// Same as SpatialCell::get_velocity_block_indices but revised for GPU. Constructs 3d indices from 1d index.
__device__ ind3d GPU_velocity_grid::get_velocity_block_indices(const unsigned int blockid) {
    ind3d indices;
    indices.x = blockid % vx_length;
    indices.y = (blockid / vx_length) % vy_length;
    indices.z = blockid / (vx_length * vy_length);

    return indices;
}

// Host version. Requires initialized SpatialCell static variables.
__host__ ind3d GPU_velocity_grid::get_velocity_block_indices_host(const unsigned int blockid) {
    ind3d indices;
    indices.x = blockid % SpatialCell::vx_length;
    indices.y = (blockid / SpatialCell::vx_length) % SpatialCell::vy_length;
    indices.z = blockid / (SpatialCell::vx_length * SpatialCell::vy_length);

    return indices;
}

// Constructs 1d index out of 3d indices
__device__ unsigned int GPU_velocity_grid::get_velocity_block(const ind3d indices) {
    return indices.x + indices.y * vx_length + indices.z * vx_length * vy_length;
}


// Same as print_blocks, but prints from a kernel
__global__ void kernel_print_blocks(GPU_velocity_grid grid) {
    unsigned int tid = blockIdx.x;
    unsigned int ind;
    ind3d indices;
    ind = grid.velocity_block_list[tid];
    indices = GPU_velocity_grid::get_velocity_block_indices(ind);
    printf("%5.0u: (%4i, %4i, %4i) %7.1f\n", ind, indices.x, indices.y, indices.z, grid.block_data[tid*WID3]);
}

// Wrapper for the kernel
__host__ void GPU_velocity_grid::k_print_blocks(void) {
    kernel_print_blocks<<<*num_blocks, 1>>>(*this);
    CUDACALL(cudaPeekAtLastError()); // Check for kernel launch errors
    CUDACALL(cudaDeviceSynchronize()); // Check for other cuda errors
}

// Prints information about transferred blocks from gpu memory
__host__ void GPU_velocity_grid::print_blocks(void) {
    printf("Number of blocks: %4u.\n", *num_blocks);
    unsigned int ind;
    ind3d indices;
    for (int i=0; i<*num_blocks; i++) {
        ind = velocity_block_list[i];
        printf("%5.0u: ", ind);
        indices = get_velocity_block_indices_host(ind);
        printf("(%4i, %4i, %4i) %7.1f\n", indices.x, indices.y, indices.z, block_data[i*WID3]);
    }
}

// Returns the data from a given block and cell id.
__device__ float GPU_velocity_grid::get_velocity_cell(unsigned int blockid, unsigned int cellid) {
    ind3d block_indices = GPU_velocity_grid::get_velocity_block_indices(blockid);
    // Check for out of bounds
    if (block_indices.x > max3d.x ||
        block_indices.y > max3d.y ||
        block_indices.z > max3d.z ||
        block_indices.x < min3d.x ||
        block_indices.y < min3d.y ||
        block_indices.z < min3d.z) return ERROR_CELL;
    vel_block *block = &vel_grid[blockid - min1d];
    return block->data[cellid];
}

// Sets the data in a given block and cell id to val. Returns the old value of the cell.
__device__ float GPU_velocity_grid::set_velocity_cell(unsigned int blockid, unsigned int cellid, float val) {
    ind3d block_indices = GPU_velocity_grid::get_velocity_block_indices(blockid);
    // Check for out of bounds
    if (block_indices.x > max3d.x ||
        block_indices.y > max3d.y ||
        block_indices.z > max3d.z ||
        block_indices.x < min3d.x ||
        block_indices.y < min3d.y ||
        block_indices.z < min3d.z) return ERROR_CELL;
    vel_block *block = &vel_grid[blockid - min1d];
    float old = block->data[cellid];
    block->data[cellid] = val;
    return old;
}

// Sets the data in a given block to that of vals.
__device__ void GPU_velocity_grid::set_velocity_block(unsigned int blockid, float *vals) {
    ind3d block_indices = GPU_velocity_grid::get_velocity_block_indices(blockid);
    // Check for out of bounds
    if (block_indices.x > max3d.x ||
        block_indices.y > max3d.y ||
        block_indices.z > max3d.z ||
        block_indices.x < min3d.x ||
        block_indices.y < min3d.y ||
        block_indices.z < min3d.z) return;
    vel_block *block = &vel_grid[blockid - min1d];
    for (int i = 0; i < WID3; i++){
        block->data[i] = vals[i];
    }
    return;
}

// Fills the given array of size len with val
__global__ void init_data(vel_block *grid, float val, int len) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
        for (int j = 0; j < WID3; j++) grid[i].data[j] = val;
    }
    __syncthreads();
}

// Copies data from block_data to vel_grid
__global__ void copy_block_data(GPU_velocity_grid *ggrid) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < *(ggrid->num_blocks)) {
        int blockid = ggrid->velocity_block_list[i];
        //ggrid->set_velocity_block(blockid, &(ggrid->block_data[i*WID3]));
    }
}

// Allocates a full velocity grid and copies data from block_data.
__host__ void GPU_velocity_grid::init_grid(void) {
    unsigned int min = this->min_ind();
    unsigned int max = this->max_ind();
    ind3d min_i = get_velocity_block_indices_host(min);
    ind3d max_i = get_velocity_block_indices_host(max);
    
    // dimensions of the grid
    unsigned int dx = max_i.x - min_i.x;
    unsigned int dy = max_i.y - min_i.y;
    unsigned int dz = max_i.z - min_i.z;
    unsigned int grid_len = dx*dy*dz;
    
    // Copy to constant memory
    CUDACALL(cudaMemcpyToSymbol(min1d, &min, sizeof(unsigned int)));
    CUDACALL(cudaMemcpyToSymbol(max1d, &max, sizeof(unsigned int)));
    CUDACALL(cudaMemcpyToSymbol(min3d, &min_i, sizeof(ind3d)));
    CUDACALL(cudaMemcpyToSymbol(max3d, &max_i, sizeof(ind3d)));
    CUDACALL(cudaMalloc(&vel_grid, grid_len * sizeof(vel_block)));
    
    // Calculate grid dimensions and start kernel
    unsigned int blockSize = 64;
    unsigned int gridSize = ceilDivide(grid_len, blockSize);
    init_data<<<gridSize, blockSize>>>(vel_grid, 0.0f, grid_len);
    cudaMemcpy(&gridSize, num_blocks, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    gridSize = ceilDivide(gridSize, blockSize);
    copy_block_data<<<gridSize, blockSize>>>(this);
}


