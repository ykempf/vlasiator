#include "gpu_velocity_grid.hpp"

using namespace spatial_cell;

// Constant memory can not be allocated inside class definition, therefore only accessed directly from this file and via accessors if necessary.
// Identical to those of SpatialCell aka. dimensions of velocity space.
__constant__ unsigned int vx_length, \
                          vy_length, \
                          vz_length;
// Minimum and maximum points of bounding box and lengths of each dimension.
__constant__ ind3d min3d, max3d, box_dims;

// Copies velocity_block_list and block_data as well as necessary constants from a SpatialCell to GPU for processing.
GPU_velocity_grid::GPU_velocity_grid(SpatialCell *spacell) {
    cpu_cell = spacell;
    // Allocate memory on the gpu
    unsigned int vel_block_list_size = spacell->number_of_blocks*sizeof(unsigned int);
    unsigned int block_data_size = spacell->block_data.size() * sizeof(Real);

    CUDACALL(cudaMalloc(&num_blocks, sizeof(unsigned int)));
    CUDACALL(cudaMalloc(&velocity_block_list, vel_block_list_size));
    CUDACALL(cudaMalloc(&block_data, block_data_size));
    CUDACALL(cudaMalloc(&min_val, sizeof(Real)));
    
    // Copy to gpu
    unsigned int *velocity_block_list_arr = &(spacell->velocity_block_list[0]);
    Real *block_data_arr = &(spacell->block_data[0]);

    num_blocks_host = spacell->number_of_blocks;
    CUDACALL(cudaMemcpy(min_val, &(SpatialCell::velocity_block_min_value), sizeof(Real), cudaMemcpyHostToDevice));
    CUDACALL(cudaMemcpy(num_blocks, &(spacell->number_of_blocks), sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDACALL(cudaMemcpyToSymbol(vx_length, &SpatialCell::vx_length, sizeof(unsigned int)));
    CUDACALL(cudaMemcpyToSymbol(vy_length, &SpatialCell::vy_length, sizeof(unsigned int)));
    CUDACALL(cudaMemcpyToSymbol(vz_length, &SpatialCell::vz_length, sizeof(unsigned int)));
    CUDACALL(cudaMemcpy(velocity_block_list, velocity_block_list_arr, vel_block_list_size, cudaMemcpyHostToDevice));
    CUDACALL(cudaMemcpy(block_data, block_data_arr, block_data_size, cudaMemcpyHostToDevice));
}

// The proper destructor for GPU_velocity_grid that has to be called manually. See the destructor comments for details.
__host__ void GPU_velocity_grid::del(void) {
// Free memory
    CUDACALL(cudaFree(num_blocks));
    CUDACALL(cudaFree(velocity_block_list));
    CUDACALL(cudaFree(block_data));
    CUDACALL(cudaFree(vel_grid));
}

// Nothing in here because this is called whenever a copy-by-value goes out of scope. Call dell when you want to free memory related to the instance.
GPU_velocity_grid::~GPU_velocity_grid() {}

__global__ void print_constants_k(void) {
    printf("vx_length: %u, vy_length: %u, vz_length: %u\n", vx_length, vy_length, vz_length);
}
void print_constants(void) {
    // Easiest to print from a kernel
    print_constants_k<<<1,1>>>();
}

__global__ void print_cells_k(GPU_velocity_grid ggrid) {
    ind3d inds = {15,15,15};
    unsigned int ind = ggrid.get_velocity_block(inds);
    printf("%u %u %u: %e \n", inds.x, inds.y, inds.z, ggrid.get_velocity_cell(ind, 0));
    inds.x = 16; inds.y = 16; inds.z = 16;
    ind = ggrid.get_velocity_block(inds);
    printf("%u %u %u: %e \n", inds.x, inds.y, inds.z, ggrid.get_velocity_cell(ind, 0));inds.x = 17; inds.y = 17; inds.z = 17;
    ind = ggrid.get_velocity_block(inds);
    printf("%u %u %u: %e \n", inds.x, inds.y, inds.z, ggrid.get_velocity_cell(ind, 0));
}
void GPU_velocity_grid::print_cells(void) {
    print_cells_k<<<1,1>>>(*this);
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

__device__ ind3d GPU_velocity_grid::get_full_grid_block_indices(const unsigned int blockid) {
    ind3d indices;
    indices.x = blockid % box_dims.x;
    indices.y = (blockid / box_dims.x) % box_dims.y;
    indices.z = blockid / (box_dims.x * box_dims.y);

    return indices;
}

__host__ ind3d GPU_velocity_grid::get_full_grid_block_indices_host(const unsigned int blockid, ind3d dims) {
    ind3d indices;
    indices.x = blockid % dims.x;
    indices.y = (blockid / dims.x) % dims.y;
    indices.z = blockid / (dims.x * dims.y);

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
    unsigned int ret = indices.x + indices.y * vx_length + indices.z * vx_length * vy_length;
    //printf("%u %u %u: %u\n", indices.x, indices.y, indices.z, ret);
    return ret;
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

__device__ vel_block* GPU_velocity_grid::get_velocity_grid_block(unsigned int blockid) {
    ind3d block_indices = GPU_velocity_grid::get_velocity_block_indices(blockid);
    //printf("%u: %u %u %u\n", blockid, block_indices.x, block_indices.y, block_indices.z);
    // Check for out of bounds
    if (block_indices.x > max3d.x ||
        block_indices.y > max3d.y ||
        block_indices.z > max3d.z ||
        block_indices.x < min3d.x ||
        block_indices.y < min3d.y ||
        block_indices.z < min3d.z) return ERROR_BLOCK;
    // Move the indices to same origin and dimensions as the bounding box
    ind3d n_ind = {block_indices.x - min3d.x, block_indices.y - min3d.y, block_indices.z - min3d.z};
    vel_block *block_ptr = &vel_grid[n_ind.x + n_ind.y*box_dims.x + n_ind.z*box_dims.x*box_dims.y];
    //printf("%4u: %2u %2u %2u, %2u %2u %2u. %016lx\n", n_ind.x + n_ind.y*box_dims.x + n_ind.z*box_dims.x*box_dims.y, block_indices.x, block_indices.y, block_indices.z, n_ind.x, n_ind.y, n_ind.z, block_ptr);
    return block_ptr;
}

// Returns index of the sparse grid corresponding to the blockid of the full grid
__device__ int GPU_velocity_grid::full_to_sparse_ind(unsigned int blockid) {
    ind3d full_inds = get_full_grid_block_indices(blockid);
    ind3d sparse_inds = {min3d.x + full_inds.x, min3d.y + full_inds.y, min3d.z + full_inds.z};
    return sparse_inds.x + sparse_inds.y * vx_length + sparse_inds.z * vx_length * vy_length;
}

// Same as above for host. Requires indices of the minimum point of the full grid.
__host__ int GPU_velocity_grid::full_to_sparse_ind_host(unsigned int blockid, ind3d dims, ind3d mins) {
        ind3d full_inds = get_full_grid_block_indices_host(blockid, dims);
        ind3d sparse_inds = {mins.x + full_inds.x, mins.y + full_inds.y, mins.z + full_inds.z};
    return sparse_inds.x + sparse_inds.y * SpatialCell::vx_length + sparse_inds.z * SpatialCell::vx_length * SpatialCell::vy_length;
}

// Returns the data from a given block and cell id.
__device__ Real GPU_velocity_grid::get_velocity_cell(unsigned int blockid, unsigned int cellid) {
    vel_block *block = get_velocity_grid_block(blockid);
    // Check for out of bounds
    if (block == ERROR_BLOCK) return ERROR_CELL;
    if (cellid >= WID3) return ERROR_CELL;
    //unsigned int indx = (*num_blocks)-5;
    //printf("%08lx ", &(vel_grid[0]));
    //printf("%08lx\n", &(vel_grid[indx]));
    //printf("%u %u %08lx\n", blockid, cellid, block->data);
    Real ret = block->data[cellid];
    return ret;
}

// Sets the data in a given block and cell id to val. Returns the old value of the cell.
__device__ Real GPU_velocity_grid::set_velocity_cell(unsigned int blockid, unsigned int cellid, Real val) {
    vel_block *block = get_velocity_grid_block(blockid);
    // Check for out of bounds
    if (block == ERROR_BLOCK) return ERROR_CELL;
    Real old = block->data[cellid];
    block->data[cellid] = val;
    return old;
}

// Sets the data in a given block to that of vals.
__device__ void GPU_velocity_grid::set_velocity_block(unsigned int blockid, Real *vals) {
    vel_block *block = get_velocity_grid_block(blockid);
    // Check for out of bounds
    if (block == ERROR_BLOCK) {
        printf("Error bad index in set_velocity_block: %u\n", blockid);
        return;
    }
    for (int i = 0; i < WID3; i++){
        block->data[i] = vals[i];
    }
    __syncthreads();
    return;
}

// Fills the given array of size len with val
__global__ void init_data(vel_block *grid, Real val, int len) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
        for (int j = 0; j < WID3; j++) {
            grid[i].data[j] = val;
        }
    }
}

// Copies data from block_data to vel_grid
__global__ void copy_block_data(GPU_velocity_grid ggrid) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < *(ggrid.num_blocks)) {
        int blockid = ggrid.velocity_block_list[i];
        ggrid.set_velocity_block(blockid, &(ggrid.block_data[i*WID3]));
    }
}

// Allocates a full velocity grid and copies data from block_data.
__host__ void GPU_velocity_grid::init_grid(void) {
    unsigned int min = this->min_ind();
    unsigned int max = this->max_ind();
    ind3d min_i = get_velocity_block_indices_host(min);
    ind3d max_i = get_velocity_block_indices_host(max);
    printf("MIN: %u %u %u %u\n", min, min_i.x, min_i.y, min_i.z);
    printf("MAX: %u %u %u %u\n", max, max_i.x, max_i.y, max_i.z);
    // dimensions of the grid
    unsigned int dx = max_i.x - min_i.x + 1;
    unsigned int dy = max_i.y - min_i.y + 1;
    unsigned int dz = max_i.z - min_i.z + 1;
    unsigned int vel_grid_len = dx*dy*dz;
    printf("GRID DIMS: %u %u %u: %u\n", dx, dy, dz, vel_grid_len);
    ind3d dims = {dx, dy, dz};
    // Copy to constant memory
    CUDACALL(cudaMemcpyToSymbol(min3d, &min_i, sizeof(ind3d)));
    CUDACALL(cudaMemcpyToSymbol(max3d, &max_i, sizeof(ind3d)));
    CUDACALL(cudaMemcpyToSymbol(box_dims, &dims, sizeof(ind3d)));
    CUDACALL(cudaMalloc(&vel_grid, vel_grid_len * sizeof(vel_block)));
    
    // Calculate grid dimensions and start kernel
    unsigned int blockSize = 64;
    unsigned int gridSize = ceilDivide(vel_grid_len, blockSize);
    init_data<<<gridSize, blockSize>>>(vel_grid, 0.0f, vel_grid_len);
    gridSize = num_blocks_host;
    printf("%u ", gridSize);
    gridSize = ceilDivide(gridSize, blockSize);
    printf("%u %u\n", gridSize, blockSize);
    CUDACALL(cudaDeviceSynchronize()); // Wait for initialization to finish
    copy_block_data<<<gridSize, blockSize>>>(*this);
    CUDACALL(cudaDeviceSynchronize()); // Block before returning
}

// Creates a list of booleans in allocated list "list" where list[i] is true if the block at vel_grid[i] includes a cell with a value larger than SpatialCell::velocity_block_min_value. N is the size of vel_grid.
__global__ void relevant_block_list(bool *list, int N, GPU_velocity_grid grid) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < N) {
        Real min_value = *grid.min_val;
        int i;
        vel_block *block_ptr = &(grid.vel_grid[tid]);
        for (i = 0; i < WID3; i++) {
            //printf("%i %i %i %016lx %016lx\n", tid, i, N, block_ptr, block_ptr->data);
            if (block_ptr->data[i] > min_value) {
                list[tid] = true;
                break;
            }
        }
        if (i == WID3) list[tid] = false;
    }
    __syncthreads();
}

// Creates a new SpatialCell with data from the full grid on GPU
__host__ SpatialCell* GPU_velocity_grid::toSpatialCell(void) {
    SpatialCell *spacell = cpu_cell; // The input SpatialCell is used to create the output.
    ind3d bounding_box_dims, bounding_box_mins;
    bool *relevant_blocks;
    CUDACALL(cudaMemcpyFromSymbol(&bounding_box_dims, box_dims, sizeof(ind3d)));
    CUDACALL(cudaMemcpyFromSymbol(&bounding_box_mins, min3d, sizeof(ind3d)));

    int box_size = bounding_box_dims.x * bounding_box_dims.y * bounding_box_dims.z;
    CUDACALL(cudaMalloc(&relevant_blocks, box_size * sizeof(bool)));
    
    const int blockSize = 64;
    const int gridSize = ceilDivide(box_size, 64);
    relevant_block_list<<<gridSize, blockSize>>>(relevant_blocks, box_size, *this);
    
    clear_data(spacell); // Remove block data but keep memory allocation. Many of the original blocks should still exist, so no need to allocate for them again.
    
    bool *rel_blocks = (bool *)malloc(box_size * sizeof(bool));
    CUDACALL(cudaDeviceSynchronize());
    CUDACALL(cudaMemcpy(rel_blocks, relevant_blocks, box_size * sizeof(bool), cudaMemcpyDeviceToHost));
    
    unsigned int ind;
    std::vector<int> rel_block_inds;
    for (int i = 0; i < box_size; i++) {
        // See if the block should be copied.
        if (!rel_blocks[i]) continue;
        ind = full_to_sparse_ind_host(i, bounding_box_dims, bounding_box_mins);
        rel_block_inds.push_back(ind);
        // Create the block in SpatialCell
        spacell->add_velocity_block(ind);
        Velocity_Block* block_ptr = spacell->at(ind);
        // Copy the data over blockwise.
        //CUDACALL(cudaMemcpyAsync(&(block_ptr->data[0]), &(vel_grid[i].data[0]), 1 * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDACALL(cudaMemcpy(&(block_ptr->data[0]), &(vel_grid[i].data[0]), WID3 * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    /*
    printf("Number of relevant blocks: %4lu\n", rel_block_inds.size());
    for (int i = 0; i < rel_block_inds.size(); i++) {
        int ind = rel_block_inds[i];
        Velocity_Block* block_ptr = spacell->at(ind);
        ind3d inds = GPU_velocity_grid::get_velocity_block_indices_host(ind);
        printf(block_print_format, ind, inds.x, inds.y, inds.z, block_ptr->data[0]);
    }
    putchar('\n');
    */
    CUDACALL(cudaFree(relevant_blocks));
    CUDACALL(cudaDeviceSynchronize());
    return spacell;
}

__global__ void print_velocity_block_list_k(GPU_velocity_grid ggrid) {
    const int cellid = 0;
    printf("ggrid:\n");
    printf("Number of relevant blocks: %4u\n", *ggrid.num_blocks);
    for (int i = 0; i < *(ggrid.num_blocks); i++) {
        int ind = ggrid.velocity_block_list[i];
        ind3d inds = GPU_velocity_grid::get_velocity_block_indices(ind);
        printf(block_print_format, ind, inds.x, inds.y, inds.z, ggrid.block_data[i*WID3 + cellid]);
    }
}

__host__ void GPU_velocity_grid::print_velocity_block_list(void) {
    print_velocity_block_list_k<<<1,1>>>(*this);
    cudaDeviceSynchronize();
}


