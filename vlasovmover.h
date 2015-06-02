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

void calculateAcceleration(
                           dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                           Real dt
);

void calculateSpatialTranslation(
                                 dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                                 Real dt);

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
void calculateCellVelocityMoments(
   SpatialCell* SC,
   bool doNotSkip=false
);


/*!
  \brief Compute 0th, 1st and 2nd velocity moments (RHO,RHOVX,RHOVY,RHOVZ,P_11,P_22,P_33 and *_DT2) for all cells in the grid directly from distribution function. The simulation should be at a true time-step! This is at the moment only called at initialisation.
  \param mpiGrid Grid of spatial cells for which moments are computed 
  
*/
void calculateInitialVelocityMoments(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid);


/*! A namespace for storing indices into an array which contains 
 * the intersections for accelerations.*/
namespace AccelerationIntersections {
   enum {
      X,    /*!< Map in x-coordinate. Intersection x-coordinate at i,j,k=0 */
      X_DI, /*!<Map in x-coordinate. Change in x-coordinate for a change in i index of 1*/
      X_DJ, /*!<Map in x-coordinate. Change in x-coordinate for a change in j index of 1*/
      X_DK, /*!<Map in x-coordinate. Change in x-coordinate for a change in k index of 1*/
      Y,    /*!< Map in y-coordinate. Intersection y-coordinate at i,j,k=0 */
      Y_DI, /*!<Map in y-coordinate. Change in y-coordinate for a change in i index of 1*/
      Y_DJ, /*!<Map in y-coordinate. Change in y-coordinate for a change in j index of 1*/
      Y_DK, /*!<Map in y-coordinate. Change in y-coordinate for a change in k index of 1*/
      Z,    /*!< Map in z-coordinate. Intersection z-coordinate at i,j,k=0 */
      Z_DI, /*!<Map in z-coordinate. Change in z-coordinate for a change in i index of 1*/
      Z_DJ, /*!<Map in z-coordinate. Change in z-coordinate for a change in j index of 1*/
      Z_DK, /*!<Map in z-coordinate. Change in z-coordinate for a change in k index of 1*/
      N_INTERSECTIONS
   };
}
         
#endif


