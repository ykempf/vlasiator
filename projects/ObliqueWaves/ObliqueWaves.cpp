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

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "../../common.h"
#include "../../readparameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../backgroundfield/constantfield.hpp"
#include "../../object_wrapper.h"


#include "ObliqueWaves.h"

using namespace std;
using namespace spatial_cell;


namespace projects {
   ObliqueWaves::ObliqueWaves(): TriAxisSearch() { }
   
   ObliqueWaves::~ObliqueWaves() { }

   bool ObliqueWaves::initialize(void) {
      return Project::initialize();
   }

   void ObliqueWaves::addParameters(){
      typedef Readparameters RP;
      RP::addComposing("ObliqueWaves.rho", "Number density (m^-3)");
      RP::addComposing("ObliqueWaves.Tx", "Temperature (K)");
      RP::addComposing("ObliqueWaves.Ty", "Temperature");
      RP::addComposing("ObliqueWaves.Tz", "Temperature");
      RP::addComposing("ObliqueWaves.Vx", "Bulk velocity x component (m/s)");
      RP::addComposing("ObliqueWaves.Vy", "Bulk velocity y component (m/s)");
      RP::addComposing("ObliqueWaves.Vz", "Bulk velocity z component (m/s)");
      RP::add("ObliqueWaves.Bx", "Magnetic field x component (T)", 0.0);
      RP::add("ObliqueWaves.By", "Magnetic field y component (T)", 0.0);
      RP::add("ObliqueWaves.Bz", "Magnetic field z component (T)", 0.0);
      RP::add("ObliqueWaves.dB", "Magnetic field cosine perturbation amplitude (T)", 0.0);
      RP::add("ObliqueWaves.lambda", "B cosine perturbation wavelength (m)", 1.0);
      RP::add("ObliqueWaves.angleDeg", "B cosine perturbation angle wrt x (degree)", 0.0);
      RP::add("ObliqueWaves.nVelocitySamples", "Number of sampling points per velocity dimension", 2);
      RP::add("ObliqueWaves.useMultipleSpecies","Is each peak a separate particle species",false);
   }

   void ObliqueWaves::getParameters(){
      typedef Readparameters RP;
      Project::getParameters();
      RP::get("ObliqueWaves.rho", this->rho);
      RP::get("ObliqueWaves.Tx", this->Tx);
      RP::get("ObliqueWaves.Ty", this->Ty);
      RP::get("ObliqueWaves.Tz", this->Tz);
      RP::get("ObliqueWaves.Vx", this->Vx);
      RP::get("ObliqueWaves.Vy", this->Vy);
      RP::get("ObliqueWaves.Vz", this->Vz);
      RP::get("ObliqueWaves.Bx", this->Bx);
      RP::get("ObliqueWaves.By", this->By);
      RP::get("ObliqueWaves.Bz", this->Bz);
      RP::get("ObliqueWaves.dB", this->dB);
      RP::get("ObliqueWaves.lambda", this->lambda);
      RP::get("ObliqueWaves.angleDeg", this->angle);
      this->angle *= 3.14159265 / 180.0;
      RP::get("ObliqueWaves.nVelocitySamples", this->nVelocitySamples);
   }

   Real ObliqueWaves::getDistribValue(creal& vx, creal& vy, creal& vz, creal& dvx, creal& dvy, creal& dvz,const int& popID) const {
      creal mass = getObjectWrapper().particleSpecies[popID].mass;
      creal kb = physicalconstants::K_B;

      Real value = 0.0;

      if (popID != 0) return 0.0;

      value = this->rho * pow(mass / (2.0 * M_PI * kb ), 1.5) * 1.0
                  / sqrt(Tx*Ty*Tz)
                  * exp(- mass * (pow(vx - Vx, 2.0) / (2.0 * kb * Tx) 
                                + pow(vy - Vy, 2.0) / (2.0 * kb * Ty) 
                                + pow(vz - Vz, 2.0) / (2.0 * kb * Tz)));
      return value;
   }

   Real ObliqueWaves::calcPhaseSpaceDensity(creal& x, creal& y, creal& z, creal& dx, creal& dy, creal& dz, 
                                         creal& vx, creal& vy, creal& vz, creal& dvx, creal& dvy, creal& dvz,
                                         const int& popID) const {
      // Iterative sampling of the distribution function. Keep track of the 
      // accumulated volume average over the iterations. When the next 
      // iteration improves the average by less than 1%, return the value.
      Real avgTotal = 0.0;
      bool ok = false;
      int N = nVelocitySamples; // Start by using nVelocitySamples
      int N3_sum = 0;           // Sum of sampling points used so far

      #warning TODO: Replace getObjectWrapper().particleSpecies[popID].sparseMinValue with SpatialCell::velocity_block_threshold(?)
      const Real avgLimit = 0.01*getObjectWrapper().particleSpecies[popID].sparseMinValue;
      do {
         Real avg = 0.0;        // Volume average obtained during this sampling
         creal DVX = dvx / N;
         creal DVY = dvy / N;
         creal DVZ = dvz / N;

         // Sample the distribution using N*N*N points
         for (uint vi=0; vi<N; ++vi) {
            for (uint vj=0; vj<N; ++vj) {
               for (uint vk=0; vk<N; ++vk) {
                  creal VX = vx + 0.5*DVX + vi*DVX;
                  creal VY = vy + 0.5*DVY + vj*DVY;
                  creal VZ = vz + 0.5*DVZ + vk*DVZ;
                  avg += getDistribValue(VX,VY,VZ,DVX,DVY,DVZ,popID);
               }
            }
         }

         // Compare the current and accumulated volume averages:
         Real eps = max(numeric_limits<creal>::min(),avg * static_cast<Real>(1e-6));
         Real avgAccum   = avgTotal / (avg + N3_sum);
         Real avgCurrent = avg / (N*N*N);
         if (fabs(avgCurrent-avgAccum)/(avgAccum+eps) < 0.01) ok = true;
         else if (avg < avgLimit) ok = true;
         else if (N > 10) {
            ok = true;
         }

         avgTotal += avg;
         N3_sum += N*N*N;
         ++N;
      } while (ok == false);

      return avgTotal / N3_sum;
   }

   void ObliqueWaves::calcCellParameters(spatial_cell::SpatialCell* cell,creal& t) {
      Real* cellParams = cell->get_cell_parameters();

      creal ang = this->angle * cellParams[CellParams::YCRD] / fabs(cellParams[CellParams::YCRD]);

      if (this->lambda != 0.0) {
         cellParams[CellParams::PERBX] = 0.0;
         cellParams[CellParams::PERBY] = this->dB*cos(2.0 * M_PI * (cos(ang)*cellParams[CellParams::XCRD] - sin(ang)*cellParams[CellParams::YCRD]) / this->lambda);
         cellParams[CellParams::PERBZ] = this->dB*sin(2.0 * M_PI * (sin(ang)*cellParams[CellParams::XCRD] + cos(ang)*cellParams[CellParams::YCRD]) / this->lambda);

      }
   }

   void ObliqueWaves::setActivePopulation(const int& popID) {
      this->popID = popID;
   }

   void ObliqueWaves::setCellBackgroundField(SpatialCell* cell) const {
      ConstantField bgField;
      bgField.initialize(this->Bx,
                         this->By,
                         this->Bz);
      setBackgroundField(bgField,cell->parameters, cell->derivatives,cell->derivativesBVOL);
   }
   
   std::vector<std::array<Real, 3> > ObliqueWaves::getV0(
                                                creal x,
                                                creal y,
                                                creal z
                                               ) const {
      vector<std::array<Real, 3> > centerPoints;
      array<Real, 3> point = {{this->Vx, this->Vy, this->Vz}};
      centerPoints.push_back(point);
      return centerPoints;
   }
   
}// namespace projects
