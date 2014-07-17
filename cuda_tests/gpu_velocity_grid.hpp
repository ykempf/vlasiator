#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <iostream>
#include "../spatial_cell.hpp"
#include "spatial_cell_funcs.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#define ERROR_CELL -1.0f
#define ERROR_BLOCK NULL

#define block_print_format "%5i(%03u,%03u,%03u)%+5.2e, "

#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU error(%i): %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Returns the the next multiple of divisor of the equivalent float division. Intended to be used with integer types as this assumes integer arithmetic.
template<typename T>
inline T ceilDivide(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// 3d indices
struct ind3d{unsigned int x,y,z;};
//typedef uint3 ind3d;
// analogous to class VelocityBlock of SpatialCell
typedef struct{Real data[WID3];} vel_block;

class GPU_velocity_grid {
    private:
        spatial_cell::SpatialCell *cpu_cell;
    public:
        unsigned int *num_blocks, num_blocks_host;
        unsigned int *velocity_block_list;
        Real *block_data;
        Real *min_val;
        vel_block *vel_grid;
        
        GPU_velocity_grid(spatial_cell::SpatialCell *spacell);
        ~GPU_velocity_grid(void); // Dummy destructor that does not do anything to make passing by value possible.
        __host__ void del(void); // The actual destructor

        __host__ void init_grid(void);
        __host__ spatial_cell::SpatialCell *toSpatialCell(void);
        __host__ unsigned int min_ind(void);
        __host__ unsigned int max_ind(void);

        // Accessor functions. The blockid here refers to the blockid in the sparse grid.
        __device__ static inline unsigned int vx_len(void);
        __device__ static inline unsigned int vy_len(void);
        __device__ static inline unsigned int vz_len(void);
        __device__ vel_block* get_velocity_grid_block(unsigned int blockid);
        __device__ int full_to_sparse_ind(unsigned int blockid);
        __host__ int full_to_sparse_ind_host(unsigned int blockid, ind3d dims, ind3d mins);
        __host__   ind3d get_full_grid_block_indices_host(const unsigned int blockid, ind3d mins);
        __device__ Real get_velocity_cell(unsigned int blockid, unsigned int cellid);
        __device__ Real set_velocity_cell(unsigned int blockid, unsigned int cellid, Real val);
        __device__ void set_velocity_block(unsigned int blockid, Real *vals);
        // Printing and helper functions
        __host__ void print_blocks(void);
        __host__ void k_print_blocks(void);
        __device__ static ind3d get_velocity_block_indices(const unsigned int blockid);
        __device__ ind3d get_full_grid_block_indices(const unsigned int blockid);
        __host__ static ind3d get_velocity_block_indices_host(const unsigned int blockid);
        __device__ static unsigned int get_velocity_block(const ind3d indices);
        __host__ void print_cells(void);
        __host__ void print_velocity_block_list(void);
};

void print_constants(void);
#endif
