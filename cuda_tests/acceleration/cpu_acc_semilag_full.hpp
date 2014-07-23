/*
  This file is part of Vlasiator.
  Copyright 2013,2014 Finnish Meteorological Institute
*/

#ifndef CPU_ACC_SEMILAG_FULL_H
#define CPU_ACC_SEMILAG__FULL_H

#ifdef ACC_SEMILAG_PLM
#define RECONSTRUCTION_ORDER 1
#endif
#ifdef ACC_SEMILAG_PPM
#define RECONSTRUCTION_ORDER 2
#endif

#include "algorithm"
#include "cmath"
#include "utility"

/*TODO - replace with standard library c++11 functions as soon as possible*/
#include "boost/array.hpp"
#include "boost/unordered_map.hpp"

#include "common.h"
#include "../../spatial_cell.hpp"

#include <Eigen/Geometry>
#include <Eigen/Core>

#include "vlasovsolver/cpu_acc_transform.hpp"
#include "vlasovsolver/cpu_acc_intersections.hpp"
#include "vlasovsolver/cpu_acc_map.hpp"

#include "map_3d.hpp"

using namespace std;
using namespace spatial_cell;

int dx, dy, dz; // Dimensions of the full grid
int min_x, min_y, min_z; // 

Real* to_full_grid(SpatialCell *spacell) {
  // First we find the bounding box of existing blocks
  min_x = SpatialCell::vx_length;
  min_y = SpatialCell::vy_length;
  min_z = SpatialCell::vz_length;
  int max_x = 0;
  int max_y = 0;
  int max_z = 0;

  // Loop over the blocks and find the minimum and maximum x,y,z-indices
  int blockid;
  velocity_block_indices_t indices;
  for (int i = 0; i < spacell->number_of_blocks; i++) {
    blockid = spacell->velocity_block_list[i];
    indices = SpatialCell::get_velocity_block_indices(blockid);
    if (indices[0] < min_x) min_x = indices[0];
    if (indices[1] < min_y) min_y = indices[1];
    if (indices[2] < min_z) min_z = indices[2];
    if (indices[0] > max_x) max_x = indices[0];
    if (indices[1] > max_y) max_y = indices[1];
    if (indices[2] > max_z) max_z = indices[2];
  }

  // Calculate the dimensions of the full grid. +1 because the box has to include both min and max.
  dx = max_x - min_x + 1;
  dy = max_y - min_y + 1;
  dz = max_z - min_z + 1;
  Real *full_grid = new Real[dx*dy*dz * WID3];

  // Initialize cell values to 0
  for (int i = 0; i < dx*dy*dz * WID3; i++) full_grid[i] = (Real)0.0;

  // Copy data to full grid
  Velocity_Block* block_ptr;
  for (int i = 0; i < spacell->number_of_blocks; i++) {
    blockid = spacell->velocity_block_list[i];
    block_ptr = spacell->at(blockid);
    indices = SpatialCell::get_velocity_block_indices(blockid);
    // Calculate indices to full grid
    int full_x = indices[0] - min_x;
    int full_y = indices[1] - min_y;
    int full_z = indices[2] - min_z;
    int blockpos = (full_x + full_y*dx + full_z*dx*dy) * WID3;
    // Copy data cell by cell
    for (int cell_i = 0; cell_i < WID3; cell_i++) {
      full_grid[blockpos + cell_i] = block_ptr->data[cell_i];
    }
  }
  return full_grid;
}

void data_to_SpatialCell(SpatialCell *spacell, Real *full_grid) {
   clear_data(spacell);
}

void cpu_accelerate_cell_(SpatialCell* spatial_cell,const Real dt) {
   double t1=MPI_Wtime();
   /*compute transform, forward in time and backward in time*/
   phiprof::start("compute-transform");
   //compute the transform performed in this acceleration
   Transform<Real,3,Affine> fwd_transform= compute_acceleration_transformation(spatial_cell,dt);
   Transform<Real,3,Affine> bwd_transform = fwd_transform.inverse();
   phiprof::stop("compute-transform");
   phiprof::start("compute-intersections");
   Real intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk;
   Real intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk;
   Real intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk;
   compute_intersections_z(spatial_cell, bwd_transform, fwd_transform,
                           intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk);
   compute_intersections_x(spatial_cell, bwd_transform, fwd_transform,
                           intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk);
   compute_intersections_y(spatial_cell, bwd_transform, fwd_transform,
                           intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk);
   phiprof::stop("compute-intersections");
   phiprof::start("compute-mapping");
   Real *full_grid = to_full_grid(spatial_cell);

   // Some debug prints
   printf("%i %i %i\n", dx, dy, dz);
   int first_ind = 103619;
   Real block_sum = 0.0;
   for (int ind = first_ind; ind < first_ind + WID3; ind++) block_sum += full_grid[ind];
   printf("%e\n", block_sum);
   
   Real v_min = SpatialCell::vz_min + min_z * SpatialCell::cell_dvz;
   for (int block_i = 0; block_i < dx; block_i++) {
      for (int block_j = 0; block_j < dy; block_j++) {
         for (int cell_i = 0; cell_i < WID; cell_i++) {
            for (int cell_j = 0; cell_j < WID; cell_j++) {
               propagate(full_grid, dz, v_min, SpatialCell::cell_dvz,
                  block_i, cell_i,block_j, cell_j,
                  intersection_z, intersection_z_di, intersection_z_dj, intersection_z_dk);
            }
         }
      }
   }
   
   //map_1d(spatial_cell, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk,2); /*< map along z*/
   //map_1d(spatial_cell, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk,0); /*< map along x*/
   //map_1d(spatial_cell, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk,1); /*< map along y*/



   free(full_grid);
   phiprof::stop("compute-mapping");
   double t2=MPI_Wtime();
   spatial_cell->parameters[CellParams::LBWEIGHTCOUNTER] += t2 - t1;
}


   

#endif

