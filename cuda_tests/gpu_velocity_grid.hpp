#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <iostream>
#include "../spatial_cell.hpp"
#include "spatial_cell_funcs.hpp"


#include <cuda_runtime.h>

#define ERROR_CELL -1.0f

#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU error(%i): %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Returns the ceiling of the equivalent float division. Intended to be used with integer types.
template<typename T>
inline T ceilDivide(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// 3d indices
typedef struct{unsigned int x,y,z;} ind3d;
// analogous to class VelocityBlock of SpatialCell
typedef struct{float data[WID3];} vel_block;

class GPU_velocity_grid {
    private:
    
    public:
        unsigned int *num_blocks;
        unsigned int *velocity_block_list;        
        float *block_data;
        vel_block *vel_grid;
        
		GPU_velocity_grid(spatial_cell::SpatialCell *spacell);
		~GPU_velocity_grid(void);
		// Accessor functions
		__device__ static inline unsigned int vx_len(void);
		__device__ static inline unsigned int vy_len(void);
		__device__ static inline unsigned int vz_len(void);
		// 
		__host__ unsigned int min_ind(void);
		__host__ unsigned int max_ind(void);
		__host__ void init_grid(void);
		__device__ float get_velocity_cell(unsigned int blockid, unsigned int cellid);
		__device__ float set_velocity_cell(unsigned int blockid, unsigned int cellid, float val);
		__device__ void set_velocity_block(unsigned int blockid, float *vals);
		// Printing and helper functions
		__host__ void print_blocks(void);
		__device__ static ind3d get_velocity_block_indices(const unsigned int blockid);
		__host__   static ind3d get_velocity_block_indices_host(const unsigned int blockid);
		__device__ static unsigned int get_velocity_block(const ind3d indices);
		__host__ void k_print_blocks(void);
};

void print_constants(void);
#endif
