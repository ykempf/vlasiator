/*
This file is part of Vlasiator.

Copyright 2011, 2012 Finnish Meteorological Institute












*/

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "../../common.h"
#include "../../readparameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../backgroundfield/constantfield.hpp"

#include "MultiPeak.h"

using namespace std;

namespace projects {
   MultiPeak::MultiPeak(): Project() { }
   MultiPeak::~MultiPeak() { }


   bool MultiPeak::initialize(void) {return true;}

   void MultiPeak::addParameters(){
      typedef Readparameters RP;
      RP::add("MultiPeak.rho1", "Number density, first peak (m^-3)", 0.0);
      RP::add("MultiPeak.rho2", "Number density, second peak (m^-3)", 0.0);
      RP::add("MultiPeak.Tx1", "Temperature, first peak (K)", 0.0);
      RP::add("MultiPeak.Tx2", "Temperature, second peak (K)", 0.0);
      RP::add("MultiPeak.Ty1", "Temperature, first peak (K)", 0.0);
      RP::add("MultiPeak.Ty2", "Temperature, second peak (K)", 0.0);
      RP::add("MultiPeak.Tz1", "Temperature, first peak (K)", 0.0);
      RP::add("MultiPeak.Tz2", "Temperature, second peak (K)", 0.0);
      RP::add("MultiPeak.Vx1", "Bulk velocity x component, first peak (m/s)", 0.0);
      RP::add("MultiPeak.Vx2", "Bulk velocity x component, second peak (m/s)", 0.0);
      RP::add("MultiPeak.Vy1", "Bulk velocity y component, first peak (m/s)", 0.0);
      RP::add("MultiPeak.Vy2", "Bulk velocity y component, second peak (m/s)", 0.0);
      RP::add("MultiPeak.Vz1", "Bulk velocity z component, first peak (m/s)", 0.0);
      RP::add("MultiPeak.Vz2", "Bulk velocity z component, second peak (m/s)", 0.0);
      RP::add("MultiPeak.Bx", "Magnetic field x component (T)", 0.0);
      RP::add("MultiPeak.By", "Magnetic field y component (T)", 0.0);
      RP::add("MultiPeak.Bz", "Magnetic field z component (T)", 0.0);
      RP::add("MultiPeak.lambda", "B perturbation wavelength (m)", 0.0);
      RP::add("MultiPeak.nVelocitySamples", "Number of sampling points per velocity dimension", 5);
   }

   void MultiPeak::getParameters(){
      typedef Readparameters RP;
      RP::get("MultiPeak.rho1", this->rho[0]);
      RP::get("MultiPeak.rho2", this->rho[1]);
      RP::get("MultiPeak.Tx1", this->Tx[0]);
      RP::get("MultiPeak.Tx2", this->Tx[1]);
      RP::get("MultiPeak.Ty1", this->Ty[0]);
      RP::get("MultiPeak.Ty2", this->Ty[1]);
      RP::get("MultiPeak.Tz1", this->Tz[0]);
      RP::get("MultiPeak.Tz2", this->Tz[1]);
      RP::get("MultiPeak.Vx1", this->Vx[0]);
      RP::get("MultiPeak.Vx2", this->Vx[1]);
      RP::get("MultiPeak.Vy1", this->Vy[0]);
      RP::get("MultiPeak.Vy2", this->Vy[1]);
      RP::get("MultiPeak.Vz1", this->Vz[0]);
      RP::get("MultiPeak.Vz2", this->Vz[1]);
      RP::get("MultiPeak.Bx", this->Bx);
      RP::get("MultiPeak.By", this->By);
      RP::get("MultiPeak.Bz", this->Bz);
      RP::get("MultiPeak.nVelocitySamples", this->nVelocitySamples);
   }

   Real MultiPeak::getDistribValue(creal& vx, creal& vy, creal& vz, creal& dvx, creal& dvy, creal& dvz) {
      creal mass = 1.67262171e-27; // m_p in kg
      creal k = 1.3806505e-23; // Boltzmann
      //  creal mu0 = 1.25663706144e-6; // mu_0
      //  creal q = 1.60217653e-19; // q_i
      //  creal gamma = 5./3.;
      
      Real value = 0.0;
      for(uint i=0; i<2; i++) {
         value += this->rho[i] * pow(mass / (2.0 * M_PI * k * sqrt(this->Tx[i]*this->Tx[i] + this->Ty[i]*this->Ty[i] + this->Tz[i]*this->Tz[i])), 1.5) *
      exp(- mass * (pow(vx - this->Vx[i], 2.0) / (2.0 * k * this->Tx[i]) + pow(vy - this->Vy[i], 2.0) / (2.0 * k * this->Ty[i]) + pow(vz - this->Vz[i], 2.0) / (2.0 * k * this->Tz[i])));
      }
      return value;
   }



   Real MultiPeak::calcPhaseSpaceDensity(creal& x, creal& y, creal& z, creal& dx, creal& dy, creal& dz, creal& vx, creal& vy, creal& vz, creal& dvx, creal& dvy, creal& dvz) {   
      creal d_vx = dvx / (this->nVelocitySamples-1);
      creal d_vy = dvy / (this->nVelocitySamples-1);
      creal d_vz = dvz / (this->nVelocitySamples-1);
      Real avg = 0.0;
   //#pragma omp parallel for collapse(6) reduction(+:avg)
      for (uint vi=0; vi<this->nVelocitySamples; ++vi)
      for (uint vj=0; vj<this->nVelocitySamples; ++vj)
         for (uint vk=0; vk<this->nVelocitySamples; ++vk)
            {
               avg += getDistribValue(vx+vi*d_vx, vy+vj*d_vy, vz+vk*d_vz, dvx, dvy, dvz);
            }
         return avg / pow(this->nVelocitySamples, 3.0);
   }



   void MultiPeak::calcCellParameters(Real* cellParams,creal& t) {
      cellParams[CellParams::PERBX   ] = 0.1*this->Bx*cos(cellParams[CellParams::XCRD] / lambda);
      cellParams[CellParams::PERBY   ] = 0.1*this->Bx*sin(cellParams[CellParams::XCRD] / lambda);
      cellParams[CellParams::PERBZ   ] = 0.1*this->Bx*cos(cellParams[CellParams::XCRD] / lambda);
   }

   void MultiPeak::setCellBackgroundField(SpatialCell* cell) {
      ConstantField bgField;
      bgField.initialize(this->Bx,
                         this->By,
                         this->Bz);
      
      setBackgroundField(bgField,cell->parameters, cell->derivatives,cell->derivativesBVOL);
   }
   
}// namespace projects
