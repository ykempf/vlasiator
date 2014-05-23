#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "../spatial_cell.hpp"

typedef thrust::device_vector<int> vel_block_indices_t;

using namespace spatial_cell;

class GPU_velocity_grid {
    private:
        unsigned int *num_blocks;
        unsigned int *gpu_velocity_block_list;
        
	public:
		GPU_velocity_grid(SpatialCell *spacell);
		//~GPU_velocity_grid(void);
		void print_blocks(void);
		vel_block_indices_t get_velocity_block_indices(const unsigned int blockid);
	
};

#endif
