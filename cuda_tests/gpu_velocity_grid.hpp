#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <cuda_runtime.h>
#include "../spatial_cell.hpp"

typedef struct{unsigned int ind[3];} vel_block_indices_t;

using namespace spatial_cell;

class GPU_velocity_grid {
    private:
        unsigned int *num_blocks;
        unsigned int *velocity_block_list;
        // Identical to those of SpatialCell aka. dimensions of velocity space.
        unsigned int *vx_length, \
                     *vy_length, \
                     *vz_length;
        float *block_data;
	public:
		GPU_velocity_grid(SpatialCell *spacell);
		//~GPU_velocity_grid(void);
		void print_blocks(void);
		vel_block_indices_t get_velocity_block_indices(const unsigned int blockid);
	
};



#endif
