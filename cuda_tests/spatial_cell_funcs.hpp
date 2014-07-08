#ifndef SPATIAL_CELL_FUNCS_HPP
#define SPATIAL_CELL_FUNCS_HPP


#include <stdlib.h>
#include <fstream>
#include "../spatial_cell.hpp"

void init_spatial_cell_static(void);
void print_blocks(spatial_cell::SpatialCell *cell);
spatial_cell::SpatialCell *create_index_test_cell(void);
spatial_cell::SpatialCell *create_maxwellian(float T, float rho);
void fprint_projection(float *projection, std::string filename);
float *xy_projection(spatial_cell::SpatialCell *spacell);
void clear_data(spatial_cell::SpatialCell *spacell);
std::vector<int>* sorted_velocity_block_list(spatial_cell::SpatialCell * spacell);
#endif
