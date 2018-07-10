/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
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
#include <cmath>

#include "../../common.h"
#include "../../readparameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../object_wrapper.h"

#include "Firehose.h"

using namespace std;

namespace projects {
   Firehose::Firehose(): Project() { }
   Firehose::~Firehose() { }
   
   bool Firehose::initialize(void) {return Project::initialize();}

   void Firehose::addParameters(){
      typedef Readparameters RP;

      RP::add("Firehose.Bx", "Magnetic field x component (T)", 0.0);
      RP::add("Firehose.By", "Magnetic field y component (T)", 0.0);
      RP::add("Firehose.Bz", "Magnetic field z component (T)", 0.0);
      RP::add("Firehose.lambda", "Initial perturbation wavelength (m)", 0.0);
      RP::add("Firehose.amp", "Initial perturbation amplitude (m)", 0.0);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;
         RP::add(pop + "_Firehose.rho1", "Number density, first peak (m^-3)", 0.0);
         RP::add(pop + "_Firehose.rho2", "Number density, second peak (m^-3)", 0.0);
         RP::add(pop + "_Firehose.Tx1", "Temperature x, first peak (K)", 0.0);
         RP::add(pop + "_Firehose.Tx2", "Temperature x, second peak (K)", 0.0);
         RP::add(pop + "_Firehose.Ty1", "Temperature y, first peak (K)", 0.0);
         RP::add(pop + "_Firehose.Ty2", "Temperature y, second peak (K)", 0.0);
         RP::add(pop + "_Firehose.Tz1", "Temperature z, first peak (K)", 0.0);
         RP::add(pop + "_Firehose.Tz2", "Temperature z, second peak (K)", 0.0);
         RP::add(pop + "_Firehose.Vx1", "Bulk velocity x component, first peak (m/s)", 0.0);
         RP::add(pop + "_Firehose.Vx2", "Bulk velocity x component, second peak (m/s)", 0.0);
         RP::add(pop + "_Firehose.Vy1", "Bulk velocity y component, first peak (m/s)", 0.0);
         RP::add(pop + "_Firehose.Vy2", "Bulk velocity y component, second peak (m/s)", 0.0);
         RP::add(pop + "_Firehose.Vz1", "Bulk velocity z component, first peak (m/s)", 0.0);
         RP::add(pop + "_Firehose.Vz2", "Bulk velocity z component, second peak (m/s)", 0.0);
         RP::add(pop + "_Firehose.nSpaceSamples", "Number of sampling points per spatial dimension", 2);
         RP::add(pop + "_Firehose.nVelocitySamples", "Number of sampling points per velocity dimension", 5);
      }
   }

   void Firehose::getParameters(){
      Project::getParameters();
      typedef Readparameters RP;
      RP::get("Firehose.Bx", this->Bx);
      RP::get("Firehose.By", this->By);
      RP::get("Firehose.Bz", this->Bz);
      RP::get("Firehose.lambda", this->lambda);
      RP::get("Firehose.amp", this->amp);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;
         FirehoseSpeciesParameters sP;
         RP::get(pop + "_Firehose.rho1", sP.rho[1]);
         RP::get(pop + "_Firehose.rho2", sP.rho[2]);
         RP::get(pop + "_Firehose.Tx1", sP.Tx[1]);
         RP::get(pop + "_Firehose.Tx2", sP.Tx[2]);
         RP::get(pop + "_Firehose.Ty1", sP.Ty[1]);
         RP::get(pop + "_Firehose.Ty2", sP.Ty[2]);
         RP::get(pop + "_Firehose.Tz1", sP.Tz[1]);
         RP::get(pop + "_Firehose.Tz2", sP.Tz[2]);
         RP::get(pop + "_Firehose.Vx1", sP.Vx[1]);
         RP::get(pop + "_Firehose.Vx2", sP.Vx[2]);
         RP::get(pop + "_Firehose.Vy1", sP.Vy[1]);
         RP::get(pop + "_Firehose.Vy2", sP.Vy[2]);
         RP::get(pop + "_Firehose.Vz1", sP.Vz[1]);
         RP::get(pop + "_Firehose.Vz2", sP.Vz[2]);
         RP::get(pop + "_Firehose.nSpaceSamples", sP.nSpaceSamples);
         RP::get(pop + "_Firehose.nVelocitySamples", sP.nVelocitySamples);

         speciesParams.push_back(sP);
      }
   }

   Real Firehose::profile(creal top, creal bottom, creal x) const {
      return top * (1.0 + this->amp*cos(2.0*M_PI*x/this->lambda));
   }

   Real Firehose::getDistribValue(
      creal& x, creal& y, creal& z,
      creal& vx, creal& vy, creal& vz,
      const uint popID
   ) const {
      const FirehoseSpeciesParameters& sP = speciesParams[popID];
      creal mass = getObjectWrapper().particleSpecies[popID].mass;
      creal kb = physicalconstants::K_B;
      
      Real Vx = profile(sP.Vx[1],sP.Vx[1], x);
      
      return
      sP.rho[1] * pow(mass / (2.0 * M_PI * kb * sP.Tx[1]), 1.5) *
      exp(- mass * (pow(vx - Vx, 2.0) / (2.0 * kb * sP.Tx[1]) +
                  pow(vy - sP.Vy[1], 2.0) / (2.0 * kb * sP.Ty[1]) +
               pow(vz - sP.Vz[1], 2.0) / (2.0 * kb * sP.Tz[1])));
   //   this->rho[2] * pow(mass / (2.0 * M_PI * kb * this->Tx[2]), 1.5) *
   //   exp(- mass * (pow(vx - this->Vx[2], 2.0) / (2.0 * kb * this->Tx[2]) + 
   //                 pow(vy - this->Vy[2], 2.0) / (2.0 * kb * this->Ty[2]) + 
   //           pow(vz - this->Vz[2], 2.0) / (2.0 * kb * this->Tz[2]))); 
   }

   Real Firehose::calcPhaseSpaceDensity(creal& x, creal& y, creal& z, creal& dx, creal& dy, creal& dz, creal& vx, creal& vy, creal& vz, creal& dvx, creal& dvy, creal& dvz,const uint popID) const {
      return sampleVelocitySpace(x, y, z, dx, dy, dz, vx, vy, vz, dvx, dvy, dvz, popID, speciesParams[popID].nSpaceSamples, speciesParams[popID].nVelocitySamples);
   }

   void Firehose::calcCellParameters(spatial_cell::SpatialCell* cell,creal& t) {
      Real* cellParams = cell->get_cell_parameters();
      cellParams[CellParams::PERBX   ] = this->Bx;
      cellParams[CellParams::PERBY   ] = this->By;
      cellParams[CellParams::PERBZ   ] = this->Bz;
   }
} // namespace projects
