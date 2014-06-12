#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "../spatial_cell.hpp"

// Start CUDA only part
#ifndef NO_CUDA
#include <cuda_runtime.h>

#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU error(%i): %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct{unsigned int ind[3];} ind3d;

class GPU_velocity_grid {
    public:
        unsigned int *num_blocks;
        unsigned int *velocity_block_list;        
        float *block_data;
        
        // Functions
		GPU_velocity_grid(spatial_cell::SpatialCell *spacell);
		~GPU_velocity_grid(void);
		__device__ static int vx_len(void);
		__device__ static int vy_len(void);
		__device__ static int vz_len(void);
		void print_blocks(void);
		__device__ static ind3d get_velocity_block_indices(const unsigned int blockid);
		__host__   static ind3d get_velocity_block_indices_host(const unsigned int blockid);
		void k_print_blocks(void);
	
};

#endif
// End CUDA only part


void init_spatial_cell_static(void);
void print_blocks(spatial_cell::SpatialCell *cell);
spatial_cell::SpatialCell *create_index_test_cell(void);
spatial_cell::SpatialCell *create_maxwellian(float T, float rho);
void fprint_projection(float *projection, std::string filename);
float *xy_projection(spatial_cell::SpatialCell *spacell);
#endif
