#ifndef SPATIAL_CELL_FUNCS_HPP
#define SPATIAL_CELL_FUNCS_HPP


#include <stdlib.h>
#include <fstream>
#include "../spatial_cell.hpp"

// Sets the data on all existing cells to zero.
inline void clear_data(spatial_cell::SpatialCell *spacell) {
    unsigned int ind;
    spatial_cell::Velocity_Block* block_ptr;
    for(int i = 0; i < spacell->number_of_blocks; i++) {
        ind = spacell->velocity_block_list[i];
        block_ptr = spacell->at(ind);
        for (int j = 0; j < WID3; j++) {
            block_ptr->data[j] = (Real)0.0;
        }
    }
}

void init_spatial_cell_static(void);
void print_blocks(spatial_cell::SpatialCell *cell);
spatial_cell::SpatialCell *create_index_test_cell(void);
spatial_cell::SpatialCell *create_maxwellian(Real T, Real rho, Real x_offset = 0.0);
spatial_cell::SpatialCell *create_cubic(const uint width, const Real value);
void fprint_projection(float *projection, std::string filename);
float *xy_projection(spatial_cell::SpatialCell *spacell);
std::vector<int>* sorted_velocity_block_list(spatial_cell::SpatialCell * spacell);
void cpu_acc_cell(spatial_cell::SpatialCell *spacell, const Real dt);
void print_column_to_file(const char *filename, spatial_cell::SpatialCell *spacell, const uint x, const uint y);
#endif
