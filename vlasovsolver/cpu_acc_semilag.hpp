/*
  This file is part of Vlasiator.
  Copyright 2013,2014 Finnish Meteorological Institute
*/

#ifndef CPU_ACC_SEMILAG_H
#define CPU_ACC_SEMILAG_H

#include "algorithm"
#include "cmath"
#include "utility"

/*TODO - replace with standard library c++11 functions*/
#include "boost/array.hpp"
#include "boost/unordered_map.hpp"

#include "common.h"
#include "spatial_cell.hpp"

#include <Eigen/Geometry>
#include <Eigen/Core>

#include "vlasovsolver/cpu_acc_transform.hpp"
#include "vlasovsolver/cpu_acc_intersections.hpp"
#include "vlasovsolver/cpu_acc_map.hpp"

using namespace std;
using namespace spatial_cell;
using namespace Eigen;


/*!
  \brief Transform velocity space in spatial cell

  Moves the distribution function in velocity space of given real
  space cell. The transform can represent and arbitrary rotation +
  translation transformation. Scaling is not yet supported.

  Based on SLICE-3D algorithm: Zerroukat, M., and T. Allen. "A
  three‐dimensional monotone and conservative semi‐Lagrangian scheme
  (SLICE‐3D) for transport problems." Quarterly Journal of the Royal
  Meteorological Society 138.667 (2012): 1640-1651.

  \param spatial_cell Spatial cell
  \param fwd_transform Transform forward in time for the distribution function
  \param population Which spatial cell population. Not yet supported.
  \param map_order In what order are the 1D mappings done. 0=XYZ, 1=YZX, 2=ZXY

*/

void cpu_transformVelocitySpace(SpatialCell* spatial_cell, Transform<Real,3,Affine> &fwd_transform, uint population, uint map_order) {
   phiprof::start("transform-velocity-space");
   /*compute transform backward in time*/
   Transform<Real,3,Affine> bwd_transform= fwd_transform.inverse();
   
   Real intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk;
   Real intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk;
   Real intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk;
   switch(map_order){
       case 0:
          phiprof::start("compute-intersections");
          //Map order XYZ
          compute_intersections_1st(spatial_cell, bwd_transform, fwd_transform, 0,
                                    intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk);
          compute_intersections_2nd(spatial_cell, bwd_transform, fwd_transform, 1,
                                    intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk);
          compute_intersections_3rd(spatial_cell, bwd_transform, fwd_transform, 2,
                                    intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk);
          phiprof::stop("compute-intersections");
          phiprof::start("compute-mapping");
          map_1d(spatial_cell, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk,0); /*< map along x*/
          map_1d(spatial_cell, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk,1); /*< map along y*/
          map_1d(spatial_cell, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk,2); /*< map along z*/
          phiprof::stop("compute-mapping");
          break;
          
       case 1:
          phiprof::start("compute-intersections");
          //Map order YZX
          compute_intersections_1st(spatial_cell, bwd_transform, fwd_transform, 1,
                                    intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk);
          compute_intersections_2nd(spatial_cell, bwd_transform, fwd_transform, 2,
                                    intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk);
          compute_intersections_3rd(spatial_cell, bwd_transform, fwd_transform, 0,
                                    intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk);
      
          phiprof::stop("compute-intersections");
          phiprof::start("compute-mapping");
          map_1d(spatial_cell, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk,1); /*< map along y*/
          map_1d(spatial_cell, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk,2); /*< map along z*/
          map_1d(spatial_cell, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk,0); /*< map along x*/
          phiprof::stop("compute-mapping");
          break;

       case 2:
          phiprof::start("compute-intersections");
          //Map order Z X Y
          compute_intersections_1st(spatial_cell, bwd_transform, fwd_transform, 2,
                                    intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk);
          compute_intersections_2nd(spatial_cell, bwd_transform, fwd_transform, 0,
                                    intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk);
          compute_intersections_3rd(spatial_cell, bwd_transform, fwd_transform, 1,
                                    intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk);
          phiprof::stop("compute-intersections");
          phiprof::start("compute-mapping");
          map_1d(spatial_cell, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk,2); /*< map along z*/
          map_1d(spatial_cell, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk,0); /*< map along x*/
          map_1d(spatial_cell, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk,1); /*< map along y*/
          phiprof::stop("compute-mapping");
          break;
   }
   phiprof::stop("transform-velocity-space");
}

/*!
  \brief Accelerate velocity space

    Accelerates the velocity space using a= q/m(v x B + E), where E is
    obtained from Ohm's law. It uses the slice-3D algorithm for the mappings.

  \sa  cpu_transformVelocitySpace
  
  \param spatial_cell Spatial cell
  \param map_order In what order are the 1D mappings done. 0=XYZ, 1=YZX, 2=ZXY
  \param dt Timestep in seconds
*/



void cpu_accelerate_cell(SpatialCell* spatial_cell, uint map_order, const Real dt) {
   double t1=MPI_Wtime();
   /*compute transform, forward in time and backward in time*/
   phiprof::start("compute-transform");
   //compute the transform performed in this acceleration
   Transform<Real,3,Affine> fwd_transform= compute_acceleration_transformation(spatial_cell,dt);
   phiprof::stop("compute-transform");

   cpu_transformVelocitySpace(spatial_cell, fwd_transform, 0,  map_order);
   
   double t2=MPI_Wtime();
   spatial_cell->parameters[CellParams::LBWEIGHTCOUNTER] += t2 - t1;

}

#endif

