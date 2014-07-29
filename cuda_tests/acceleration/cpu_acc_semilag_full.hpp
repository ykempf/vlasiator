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

struct full_grid_t {
  Real *grid;
  int dx, dy, dz; // Dimensions of the full grid in velocity blocks.
  int min_x, min_y, min_z; // The indices of the minimum corner of bounding box in the sparse grid.
};


full_grid_t* to_full_grid(SpatialCell *spacell) {
  full_grid_t *full_grid = new full_grid_t;
  // First we find the bounding box of existing blocks
  full_grid->min_x = SpatialCell::vx_length;
  full_grid->min_y = SpatialCell::vy_length;
  full_grid->min_z = SpatialCell::vz_length;
  // Temporary variables for maximum indices. The width and min indices are stored in full_grid struct.
  int max_x = 0;
  int max_y = 0;
  int max_z = 0;

  // Loop over the blocks and find the minimum and maximum x,y,z-indices
  int blockid;
  velocity_block_indices_t indices;
  for (int i = 0; i < spacell->number_of_blocks; i++) {
    blockid = spacell->velocity_block_list[i];
    indices = SpatialCell::get_velocity_block_indices(blockid);
    if (indices[0] < full_grid->min_x) full_grid->min_x = indices[0];
    if (indices[1] < full_grid->min_y) full_grid->min_y = indices[1];
    if (indices[2] < full_grid->min_z) full_grid->min_z = indices[2];
    if (indices[0] > max_x) max_x = indices[0];
    if (indices[1] > max_y) max_y = indices[1];
    if (indices[2] > max_z) max_z = indices[2];
  }

  // Calculate the dimensions of the full grid. +1 because the box has to include both min and max.
  full_grid->dx = max_x - full_grid->min_x + 1;
  full_grid->dy = max_y - full_grid->min_y + 1;
  full_grid->dz = max_z - full_grid->min_z + 1;
  full_grid->grid = new Real[full_grid->dx*full_grid->dy*full_grid->dz * WID3];

  // Initialize cell values to 0
  for (int i = 0; i < full_grid->dx*full_grid->dy*full_grid->dz * WID3; i++) full_grid->grid[i] = (Real)0.0;

  // Copy data to full grid
  Velocity_Block* block_ptr;
  for (int i = 0; i < spacell->number_of_blocks; i++) {
    blockid = spacell->velocity_block_list[i];
    block_ptr = spacell->at(blockid);
    indices = SpatialCell::get_velocity_block_indices(blockid);
    // Calculate indices to full grid
    int full_x = indices[0] - full_grid->min_x;
    int full_y = indices[1] - full_grid->min_y;
    int full_z = indices[2] - full_grid->min_z;
    int blockpos = (full_x + full_y*full_grid->dx + full_z*full_grid->dx*full_grid->dy) * WID3;
    //if (i==0) printf("first ind %i\n", blockpos);
    // Copy data cell by cell
    for (int cell_i = 0; cell_i < WID3; cell_i++) {
      full_grid->grid[blockpos + cell_i] = block_ptr->data[cell_i];
    }
  }
  return full_grid;
}

static int relevant_blocks = 0;

void data_to_SpatialCell(SpatialCell *spacell, full_grid_t *full_grid) {
  printf("data_to_SpatialCell:\n");
  clear_data(spacell);
  bool relevant_block;
  Real minval = SpatialCell::velocity_block_min_value;
  printf("minval : %e\n", minval);
  // Loop over blocks
  for (int block_k = 0; block_k < full_grid->dz; block_k++) {
    for (int block_j = 0; block_j < full_grid->dy; block_j++) {
      for (int block_i = 0; block_i < full_grid->dx; block_i++) {
        relevant_block = false;
        //Check if block contains relevant data
        int full_block_ind = block_i + block_j*full_grid->dx + block_k*full_grid->dx*full_grid->dy;
        for (int cell_i = 0; cell_i < WID3; cell_i++) {
          //printf("%e, %e\n", full_grid[(block_i + block_j*dx + block_k*dx*dy)*WID3 + cell_i], minval);
          if (full_grid->grid[full_block_ind*WID3 + cell_i] > minval) {
            relevant_block = true;
            break;
          }
        }
        if (relevant_block) {
          relevant_blocks++;
          // Construct index to sparse grid
          int ind = (full_grid->min_x + block_i) + (full_grid->min_y + block_j)*SpatialCell::vx_length
                    + (full_grid->min_z + block_k)*SpatialCell::vx_length*SpatialCell::vy_length;
          spacell->add_velocity_block(ind);
          Velocity_Block* block_ptr = spacell->at(ind);
          if (block_ptr == error_velocity_block || block_ptr == NULL) {
            printf("Error block: %i\n", ind);
          }
          for (int cell_i = 0 ; cell_i < WID3; cell_i++) {
            block_ptr->data[cell_i] = full_grid->grid[full_block_ind*WID3 + cell_i];
          }
        }
      }
    }
  }
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
   full_grid_t *full_grid = to_full_grid(spatial_cell);

   // Some debug prints
   printf("%i %i %i\n", full_grid->dx, full_grid->dy, full_grid->dz);
   int first_ind = 64256;
   Real block_sum = 0.0;
   for (int ind = first_ind; ind < first_ind + WID3; ind++) block_sum += full_grid->grid[ind];
   printf("%e\n", block_sum);

   Real v_min = SpatialCell::vz_min + full_grid->min_z * SpatialCell::cell_dvz;
   int column_size = full_grid->dz*WID; // In cells
   Real *column_data = new Real[column_size + 2*WID]; // propagate needs the extra cells
   Real *target_column_data;
   for (int block_i = 0; block_i < full_grid->dx; block_i++) {
      for (int block_j = 0; block_j < full_grid->dy; block_j++) {
        int blockid = block_i + block_j * full_grid->dx;
         for (int cell_i = 0; cell_i < WID; cell_i++) {
            for (int cell_j = 0; cell_j < WID; cell_j++) {
              int cellid = cell_i + cell_j*WID;
              // Construct a temporary array with only data from one column of velocity CELLS
              for (int block_k = 0; block_k < full_grid->dz; block_k++) {
                for (int cell_k = 0; cell_k < WID; ++cell_k)
                {
                  column_data[block_k*WID + cell_k + WID] = full_grid->grid[blockid*WID3 + cellid + cell_k*WID2]; // Cells in the same k column in a block are WID2 apart
                }
              }
              propagate(column_data, target_column_data, full_grid->dz, v_min, SpatialCell::cell_dvz,
                block_i, cell_i,block_j, cell_j,
                intersection_z, intersection_z_di, intersection_z_dj, intersection_z_dk);
              // Copy back to full grid
              for (int block_k = 0; block_k < full_grid->dz; block_k++) {
                for (int cell_k = 0; cell_k < WID; ++cell_k)
                {
                  full_grid->grid[blockid*WID3 + cellid + cell_k*WID2] = column_data[block_k*WID + cell_k + WID];
                }
              }
            }
         }
      } 
   }
   delete[] column_data;


   
   first_ind = 64256;
   block_sum = 0.0;
   for (int ind = first_ind; ind < first_ind + WID3; ind++) block_sum += full_grid->grid[ind];
   printf("%e\n", block_sum);

   map_1d(spatial_cell, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk,2); /*< map along z*/
   //map_1d(spatial_cell, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk,0); /*< map along x*/
   //map_1d(spatial_cell, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk,1); /*< map along y*/

   //data_to_SpatialCell(spatial_cell, full_grid);
   printf("rel blocks %i\n", relevant_blocks);
   
   /*
   // Remove unnecessary blocks
   std::vector<SpatialCell*> neighbor_ptrs;
   spatial_cell->update_velocity_block_content_lists();
   spatial_cell->adjust_velocity_blocks(neighbor_ptrs,true);
   */

   phiprof::stop("compute-mapping");
   double t2=MPI_Wtime();
   spatial_cell->parameters[CellParams::LBWEIGHTCOUNTER] += t2 - t1;
}


   

#endif

