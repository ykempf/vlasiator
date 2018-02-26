/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://vlasiator.fmi.fi/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef OBLIQUEWAVES_H
#define OBLIQUEWAVES_H

#include <vector>

#include "../../definitions.h"
#include "../projectTriAxisSearch.h"

namespace projects {
   class ObliqueWaves: public TriAxisSearch {
    public:
      ObliqueWaves();
      virtual ~ObliqueWaves();
      
      virtual bool initialize(void);
      static void addParameters(void);
      virtual void getParameters(void);
      virtual void setActivePopulation(const int& popID);
      virtual void setCellBackgroundField(spatial_cell::SpatialCell* cell) const;
    protected:
      Real getDistribValue(
                           creal& x,creal& y, creal& z,
                           creal& vx, creal& vy, creal& vz,
                          const int& popID) const;
      virtual void calcCellParameters(spatial_cell::SpatialCell* cell,creal& t);
      virtual Real calcPhaseSpaceDensity(
                                         creal& x, creal& y, creal& z,
                                         creal& dx, creal& dy, creal& dz,
                                         creal& vx, creal& vy, creal& vz,
                                         creal& dvx, creal& dvy, creal& dvz,
                                         const int& popID) const;
      virtual std::vector<std::array<Real, 3> > getV0(
                                                      creal x,
                                                      creal y,
                                                      creal z
                                                     ) const;
      int popID;
      Real rho;
      Real Tx;
      Real Ty;
      Real Tz;
      Real Vx;
      Real Vy;
      Real Vz;
      Real Bx;
      Real By;
      Real Bz;
      Real dB;
      Real lambda;
      Real angle;
      uint nVelocitySamples;

   }; // class ObliqueWaves
} //  namespace projects

#endif

