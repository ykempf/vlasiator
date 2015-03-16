/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2010, 2011, 2012, 2013 Finnish Meteorological Institute
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

/*!\file reflecting.cpp
 * \brief Implementation of the class SysBoundaryCondition::Reflecting to handle cells classified as sysboundarytype::REFLECTING.
 */

#include <cstdlib>
#include <iostream>

#include "reflecting.h"
#include "../vlasovmover.h"
#include "../projects/projects_common.h"
#include "../fieldsolver/fs_common.h"

using namespace std;

namespace SBC {
   Reflecting::Reflecting(): SysBoundaryCondition() { }
   Reflecting::~Reflecting() { }
   
   void Reflecting::addParameters() {
      Readparameters::addComposing("reflecting.face", "List of faces on which reflecting boundary conditions are to be applied ([xyz][+-]).");
      Readparameters::add("reflecting.precedence", "Precedence value of the reflecting system boundary condition (integer), the higher the stronger.", 4);
   }
   
   void Reflecting::getParameters() {
      int myRank;
      MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
      if(!Readparameters::get("reflecting.face", this->faceList)) {
         if(myRank == MASTER_RANK) cerr << __FILE__ << ":" << __LINE__ << " ERROR: This option has not been added!" << endl;
         exit(1);
      }
      if(!Readparameters::get("reflecting.precedence", this->precedence)) {
         if(myRank == MASTER_RANK) cerr << __FILE__ << ":" << __LINE__ << " ERROR: This option has not been added!" << endl;
         exit(1);
      }
   }
   
   bool Reflecting::initSysBoundary(
      creal& t,
      Project &project
   ) {
      /* The array of bool describes which of the x+, x-, y+, y-, z+, z- faces are to have reflecting system boundary conditions.
       * A true indicates the corresponding face will have reflecting.
       * The 6 elements correspond to x+, x-, y+, y-, z+, z- respectively.
       */
      for(uint i=0; i<6; i++) facesToProcess[i] = false;
      
      this->getParameters();
      
      isThisDynamic = false;
      
      vector<string>::const_iterator it;
      for (it = this->faceList.begin();
           it != this->faceList.end();
      it++) {
         if(*it == "x+") facesToProcess[0] = true;
         if(*it == "x-") facesToProcess[1] = true;
         if(*it == "y+") facesToProcess[2] = true;
         if(*it == "y-") facesToProcess[3] = true;
         if(*it == "z+") facesToProcess[4] = true;
         if(*it == "z-") facesToProcess[5] = true;
      }
      return true;
   }
   
   bool Reflecting::assignSysBoundary(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
      vector<CellID> cells = mpiGrid.get_cells();
      for(uint i = 0; i < cells.size(); i++) {
         if(mpiGrid[cells[i]]->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) continue;
         creal* const cellParams = &(mpiGrid[cells[i]]->parameters[0]);
         creal dx = cellParams[CellParams::DX];
         creal dy = cellParams[CellParams::DY];
         creal dz = cellParams[CellParams::DZ];
         creal x = cellParams[CellParams::XCRD] + 0.5*dx;
         creal y = cellParams[CellParams::YCRD] + 0.5*dy;
         creal z = cellParams[CellParams::ZCRD] + 0.5*dz;
         
         bool isThisCellOnAFace[6];
         determineFace(&isThisCellOnAFace[0], x, y, z, dx, dy, dz);
         
         // Comparison of the array defining which faces to use and the array telling on which faces this cell is
         bool doAssign = false;
         for(uint j=0; j<6; j++) doAssign = doAssign || (facesToProcess[j] && isThisCellOnAFace[j]);
         if(doAssign) {
            mpiGrid[cells[i]]->sysBoundaryFlag = this->getIndex();
         }
      }
      return true;
   }
   
   bool Reflecting::applyInitialState(
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      Project &project
   ) {
      vector<uint64_t> cells = mpiGrid.get_cells();
#pragma omp parallel for
      for (uint i=0; i<cells.size(); ++i) {
         SpatialCell* cell = mpiGrid[cells[i]];
         if(cell->sysBoundaryFlag != this->getIndex()) continue;
         
         // Defined in project.cpp, used here as the reflecting cell has the same state as the initial state of non-system boundary cells.
         project.setCell(cell);
         // WARNING Time-independence assumed here.
         cell->parameters[CellParams::RHO_DT2] = cell->parameters[CellParams::RHO];
         cell->parameters[CellParams::RHOVX_DT2] = cell->parameters[CellParams::RHOVX];
         cell->parameters[CellParams::RHOVY_DT2] = cell->parameters[CellParams::RHOVY];
         cell->parameters[CellParams::RHOVZ_DT2] = cell->parameters[CellParams::RHOVZ];
      }
      
      return true;
   }
   
   /*! We want here to
    * 
    * -- Average perturbed face B from the nearest neighbours
    * 
    * -- Retain only the normal components of perturbed face B
    */
   Real Reflecting::fieldSolverBoundaryCondMagneticField(
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,
      creal& dt,
      cuint& component
   ) {
      std::vector<CellID> closestCells = getAllClosestNonsysboundaryCells(mpiGrid, cellID);
      if (closestCells.size() == 1 && closestCells[0] == INVALID_CELLID) {
         std::cerr << __FILE__ << ":" << __LINE__ << ":" << "No closest cells found!" << std::endl;
         abort();
      }
      
      // Sum perturbed B component over all nearest NOT_SYSBOUNDARY neighbours
      std::array<Real, 3> averageB = {{ 0.0 }};
      int offset;
      if (dt == 0.0) {
         offset = 0;
      } else {
         offset = CellParams::PERBX_DT2 - CellParams::PERBX;
      }
      for(uint i=0; i<closestCells.size(); i++) {
         averageB[0] += mpiGrid[closestCells[i]]->parameters[CellParams::PERBX+offset];
         averageB[1] += mpiGrid[closestCells[i]]->parameters[CellParams::PERBY+offset];
         averageB[2] += mpiGrid[closestCells[i]]->parameters[CellParams::PERBZ+offset];
      }
      
      // Average and project to normal direction
      std::array<Real, 3> normalDirection = fieldSolverGetNormalDirection(mpiGrid, cellID);
      for(uint i=0; i<3; i++) {
         averageB[i] *= normalDirection[i] / closestCells.size();
      }
      
      // Return (B.n)*normalVector[component]
      return (averageB[0]+averageB[1]+averageB[2])*normalDirection[component];
   }
   
   void Reflecting::fieldSolverBoundaryCondElectricField(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,
      cuint RKCase,
      cuint component
   ) {
      if((RKCase == RK_ORDER1) || (RKCase == RK_ORDER2_STEP2)) {
         mpiGrid[cellID]->parameters[CellParams::EX+component] = 0.0;
      } else {// RKCase == RK_ORDER2_STEP1
         mpiGrid[cellID]->parameters[CellParams::EX_DT2+component] = 0.0;
      }
   }
   
   void Reflecting::fieldSolverBoundaryCondHallElectricField(
      dccrg::Dccrg<spatial_cell::SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,
      cuint RKCase,
      cuint component
   ) {
      switch(component) {
         case 0:
            mpiGrid[cellID]->parameters[CellParams::EXHALL_000_100] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EXHALL_010_110] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EXHALL_001_101] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EXHALL_011_111] = 0.0;
            break;
         case 1:
            mpiGrid[cellID]->parameters[CellParams::EYHALL_000_010] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EYHALL_100_110] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EYHALL_001_011] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EYHALL_101_111] = 0.0;
            break;
         case 2:
            mpiGrid[cellID]->parameters[CellParams::EZHALL_000_001] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EZHALL_100_101] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EZHALL_010_011] = 0.0;
            mpiGrid[cellID]->parameters[CellParams::EZHALL_110_111] = 0.0;
            break;
         default:
            cerr << __FILE__ << ":" << __LINE__ << ":" << " Invalid component" << endl;
      }
   }
   
   void Reflecting::fieldSolverBoundaryCondDerivatives(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,
      cuint& RKCase,
      cuint& component
   ) {
      this->setCellDerivativesToZero(mpiGrid, cellID, component);
   }
   
   void Reflecting::fieldSolverBoundaryCondBVOLDerivatives(
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,
      cuint& component
   ) {
      this->setCellBVOLDerivativesToZero(mpiGrid, cellID, component);
   }
   
   void Reflecting::vlasovBoundaryCondition(
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID
   ) {
      std::array<Real, 3> normalDirection = fieldSolverGetNormalDirection(mpiGrid, cellID);
      
      vlasovBoundaryReflect(
         mpiGrid,
         cellID,
         normalDirection[0],
         normalDirection[1],
         normalDirection[2]
      );
      
      calculateCellVelocityMoments(mpiGrid[cellID]);
      
   }
   
   std::array<Real, 3> Reflecting::fieldSolverGetNormalDirection(
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID
   ) {
      std::array<Real, 3> normalDirection{{ 0.0, 0.0, 0.0 }};
      
      creal* const cellParams = &(mpiGrid[cellID]->parameters[0]);
      creal dx = cellParams[CellParams::DX];
      creal dy = cellParams[CellParams::DY];
      creal dz = cellParams[CellParams::DZ];
      creal x = cellParams[CellParams::XCRD] + 0.5*dx;
      creal y = cellParams[CellParams::YCRD] + 0.5*dy;
      creal z = cellParams[CellParams::ZCRD] + 0.5*dz;
      
      bool isThisCellOnAFace[6];
      determineFace(&isThisCellOnAFace[0], x, y, z, dx, dy, dz);
      
      // WARNING this sets precedence but we don't want to bother with reflecting corners at the moment.
      if(isThisCellOnAFace[0] && facesToProcess[0]) {
         normalDirection[0] = -1.0;
      } else if(isThisCellOnAFace[1] && facesToProcess[1]) {
         normalDirection[0] = 1.0;
      } else if(isThisCellOnAFace[2] && facesToProcess[2]) {
         normalDirection[1] = -1.0;
      } else if(isThisCellOnAFace[3] && facesToProcess[3]) {
         normalDirection[1] = 1.0;
      } else if(isThisCellOnAFace[4] && facesToProcess[4]) {
         normalDirection[2] = -1.0;
      } else if(isThisCellOnAFace[5] && facesToProcess[5]) {
         normalDirection[2] = 1.0;
      }
      return normalDirection;
   }
   
   void Reflecting::getFaces(bool* faces) {
      for(uint i=0; i<6; i++) faces[i] = facesToProcess[i];
   }
   
   std::string Reflecting::getName() const {return "Reflecting";}
   uint Reflecting::getIndex() const {return sysboundarytype::REFLECTING;}
}
