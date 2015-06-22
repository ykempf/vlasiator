/*
This file is part of Vlasiator.

Copyright 2010, 2011, 2012, 2013 Finnish Meteorological Institute
*/

#ifndef VLASOVMOVER_H
#define VLASOVMOVER_H

#include <vector>

#include "definitions.h"
#include "spatial_cell.hpp"
using namespace spatial_cell;

#include <stdint.h>
#include <dccrg.hpp>
#include <dccrg_cartesian_geometry.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
void calculateAcceleration(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                           Real dt);

void calculateSpatialTranslation(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                                 Real dt);


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

void cpu_transformVelocitySpace(SpatialCell* spatial_cell, Eigen::Transform<Real,3,Eigen::Affine> &fwd_transform, uint population, uint map_order = 0);




/*!
  \brief Compute real-time 1st order accurate moments from the moments after propagation in velocity and spatial space
*/
 
void calculateInterpolatedVelocityMoments(
   dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   const int cp_rho,
   const int cp_rhovx,
   const int cp_rhovy,
   const int cp_rhovz,
   const int cp_p11,
   const int cp_p22,
   const int cp_p33
);

/*!
   \brief Compute 0th, 1st and 2nd velocity moments (RHO,RHOVX,RHOVY,RHOVZ,P_11,P_22,P_33) for a cell directly from distribution function. The simulation should be at a true time-step!
   \param SC pointer to the spatial cell
   \param doNotSkip Used to override the checks about system boundaries which in some cases cause the moments not to be calculated as they have been e.g. copied. Default value: false, in order to preserve all the calls using the checks.
*/
void calculateCellVelocityMoments(SpatialCell* SC, bool doNotSkip=false);


/*!
  \brief Compute 0th, 1st and 2nd velocity moments (RHO,RHOVX,RHOVY,RHOVZ,P_11,P_22,P_33 and *_DT2) for all cells in the grid directly from distribution function. The simulation should be at a true time-step! This is at the moment only called at initialisation.
  \param mpiGrid Grid of spatial cells for which moments are computed 
  
*/
void calculateInitialVelocityMoments(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid);



#endif


