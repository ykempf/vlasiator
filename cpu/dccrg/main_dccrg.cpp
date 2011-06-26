#ifndef PARGRID // Do not use these functions if ParGrid is used:

#include <cstdlib>
#include <iostream>
#include <vector>
#include <boost/mpi.hpp>
#include <zoltan.h>

#define DCCRG_SEND_SINGLE_CELLS
#define DCCRG_CELL_DATA_SIZE_FROM_USER
#define DCCRG_USER_MPI_DATA_TYPE
#include <dccrg.hpp>

#include <mpilogger.h>
#include <definitions.h>
#include <parameters.h>
#include <cell_spatial.h>
#include <project.h>
#include "vlasovmover.h"
#include <profile.h>

extern MPILogger mpilogger;

extern bool cpu_acceleration(SpatialCell& cell);
extern bool cpu_translation1(SpatialCell& cell,const std::vector<const SpatialCell*>& nbrPtrs);
extern bool cpu_translation2(SpatialCell& cell,const std::vector<const SpatialCell*>& nbrPtrs);
extern bool cpu_translation3(SpatialCell& cell,const std::vector<const SpatialCell*>& nbrPtrs);
extern bool cpu_calcVelocityMoments(SpatialCell& cell);

namespace Main {
   std::vector<uint64_t> cells;
   std::vector<const SpatialCell*> nbrPtrs(6,NULL);
   SpatialCell* cellPtr;
   
   uint calcAcc;
   uint calcSpatDerivs;
   uint spatDerivsMPIRecv;
   uint spatDerivsMPISend;
   uint calcSpatFluxes;
   uint spatFluxesMPIRecv;
   uint spatFluxesMPISend;
   uint calcSpatProp;
   uint spatPropMPIRecv;
   uint spatPropMPISend;
}

bool finalizeMover() {
   return true;
}

bool initializeMover(dccrg<SpatialCell>& mpiGrid) {
   #warning Spatial neighbour lists not populated for dccrg!   
   return true;
}

void initialLoadBalance(dccrg<SpatialCell>& mpiGrid) {
   /*typedef Parameters P;
   P::transmit = Transmit::AVGS;*/
   mpiGrid.balance_load();
}

/*!
Fills nbrPtr with pointers to data in neighbour cells or NULL if neighbour doesn't exist.
*/
bool findNeighbours(std::vector<const SpatialCell*>& nbrPtr,dccrg<SpatialCell>& mpiGrid,const uint64_t& cell) {

   const std::vector<uint64_t>* neighbours = mpiGrid.get_neighbours(cell);
   int index = 0;

   // neighbours are in a certain order in the neighbour list, -z first
   if ((*neighbours)[index] > 0) {
      nbrPtr[4] = mpiGrid[(*neighbours)[index]];
   } else {
      nbrPtr[4] = NULL;
   }
   index++;

   // -y
   if ((*neighbours)[index] > 0) {
      nbrPtr[2] = mpiGrid[(*neighbours)[index]];
   } else {
      nbrPtr[2] = NULL;
   }
   index++;
   // -x
   if ((*neighbours)[index] > 0) {
      nbrPtr[0] = mpiGrid[(*neighbours)[index]];
   } else {
      nbrPtr[0] = NULL;
   }
   index++;

   // +x
   if ((*neighbours)[index] > 0) {
      nbrPtr[1] = mpiGrid[(*neighbours)[index]];
   } else {
      nbrPtr[1] = NULL;
   }
   index++;

   // +y
   if ((*neighbours)[index] > 0) {
      nbrPtr[3] = mpiGrid[(*neighbours)[index]];
   } else {
      nbrPtr[3] = NULL;
   }
   index++;

   // +z
   if ((*neighbours)[index] > 0) {
      nbrPtr[5] = mpiGrid[(*neighbours)[index]];
   } else {
      nbrPtr[5] = NULL;
   }

   return true;
}

void calculateVelocityMoments(dccrg<SpatialCell>& mpiGrid) {
   Main::cells = mpiGrid.get_cells();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (Main::cellPtr != NULL) cpu_calcVelocityMoments(*Main::cellPtr);
   }
}

void calculateSimParameters(dccrg<SpatialCell>& mpiGrid, creal& t, Real& dt) {
   // TODO let the project function decide if something should really be calculated
   if (!cellParametersChanged(t)) {
   	return;
   }
   calcSimParameters(mpiGrid, t, dt);
}

void calculateCellParameters(dccrg<SpatialCell>& mpiGrid,creal& t, uint64_t cell) {
   // TODO let the project function decide if something should really be calculated
   if (!cellParametersChanged(t)) {
   	return;
   }
   calcCellParameters(mpiGrid[cell]->cpu_cellParams,t);
}

void calculateAcceleration(dccrg<SpatialCell>& mpiGrid) {
   profile::start("calcAcceleration");
   
   // Calculate acceleration for all cells (inner + boundary):
   Main::cells = mpiGrid.get_cells();
   for (size_t i=0; i<Main::cells.size(); ++i) {
      Main::cellPtr = mpiGrid[Main::cells[i]];
      if (Main::cellPtr != NULL) cpu_acceleration(*Main::cellPtr);
   }
   
   profile::stop("calcAcceleration",Main::cells.size(),"SpatialCells");
}

void calculateSpatialDerivatives(dccrg<SpatialCell>& mpiGrid) {
   profile::start("calcSpatDerivatives");
   profile::start("Start data exchange");
   unsigned int computedCells;
   typedef Parameters P;
   // Start neighbour data exchange:
   P::transmit = Transmit::AVGS;
   SpatialCell::base_address_identifier = 0;
   mpiGrid.start_remote_neighbour_data_update();
   profile::stop("Start data exchange");
   profile::start("Compute inner cells");
   // Calculate derivatives for inner cells:
   Main::cells = mpiGrid.get_cells_with_local_neighbours();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (findNeighbours(Main::nbrPtrs,mpiGrid,Main::cells[c]) == false) {
	 mpilogger << "Failed to find neighbours." << std::endl << write; 
	 continue;
      }
      if (Main::cellPtr != NULL) cpu_translation1(*Main::cellPtr,Main::nbrPtrs);
   }
   profile::stop("Compute inner cells",Main::cells.size(),"SpatialCells");
   computedCells=Main::cells.size();
   profile::start("Wait for receives");
   // Calculate derivatives for boundary cells when transfers have completed:
   mpiGrid.wait_neighbour_data_update_receives();
   profile::stop("Wait for receives");
   profile::start("Compute border cells");
   Main::cells = mpiGrid.get_cells_with_remote_neighbour();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (findNeighbours(Main::nbrPtrs,mpiGrid,Main::cells[c]) == false) {
	 mpilogger << "Failed to find neighbours." << std::endl << write; 
	 continue;
      }
      if (Main::cellPtr != NULL) cpu_translation1(*Main::cellPtr,Main::nbrPtrs);
   }
   profile::stop("Compute border cells",Main::cells.size(),"SpatialCells");
   computedCells+=Main::cells.size();
   profile::start("Wait for sends");
   mpiGrid.wait_neighbour_data_update_sends();
   profile::stop("Wait for sends");
   profile::stop("calcSpatDerivatives",computedCells,"SpatialCells");
}

void calculateSpatialFluxes(dccrg<SpatialCell>& mpiGrid) {
   profile::start("calcSpatFluxes");
   profile::start("Start data exchange");
   unsigned int computedCells;
   typedef Parameters P;
   // Start neighbour data exchange:
   P::transmit = Transmit::DERIV1;
   SpatialCell::base_address_identifier = 1;
   mpiGrid.start_remote_neighbour_data_update();
   profile::stop("Start data exchange");
   profile::start("Compute inner cells");
   // Calculate fluxes for inner cells:
   Main::cells = mpiGrid.get_cells_with_local_neighbours();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (findNeighbours(Main::nbrPtrs,mpiGrid,Main::cells[c]) == false) {
	 mpilogger << "Failed to find neighbours." << std::endl << write; 
	 continue;
      }
      if (Main::cellPtr != NULL) cpu_translation2(*Main::cellPtr,Main::nbrPtrs);
   }
   // Calculate fluxes for boundary cells when transfers have completed:
   profile::stop("Compute inner cells",Main::cells.size(),"SpatialCells");
   computedCells=Main::cells.size();
   profile::start("Wait for receives");
   mpiGrid.wait_neighbour_data_update_receives();
   profile::stop("Wait for receives");
   profile::start("Compute border cells"); 
   Main::cells = mpiGrid.get_cells_with_remote_neighbour();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (findNeighbours(Main::nbrPtrs,mpiGrid,Main::cells[c]) == false) {
	 mpilogger << "Failed to find neighbours." << std::endl << write; 
	 continue;
      }
      if (Main::cellPtr != NULL) cpu_translation2(*Main::cellPtr,Main::nbrPtrs);
   }
   profile::stop("Compute border cells",Main::cells.size(),"SpatialCells");
   computedCells+=Main::cells.size();
   profile::start("Wait for sends");
   mpiGrid.wait_neighbour_data_update_sends();
   profile::stop("Wait for sends");
   profile::stop("calcSpatFluxes",computedCells,"SpatialCells");
}

void calculateSpatialPropagation(dccrg<SpatialCell>& mpiGrid,const bool& secondStep,const bool& transferAvgs) {
   profile::start("calcSpatProp");
   profile::start("Start data exchange");
   unsigned int computedCells;

   typedef Parameters P;
   // Start neighbour data exchange:
   P::transmit = Transmit::FLUXES;
   SpatialCell::base_address_identifier = 2;
   mpiGrid.start_remote_neighbour_data_update();
   profile::stop("Start data exchange");
   profile::start("Compute inner cells");
   // Propagate inner cells:
   Main::cells = mpiGrid.get_cells_with_local_neighbours();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (findNeighbours(Main::nbrPtrs,mpiGrid,Main::cells[c]) == false) {
	 mpilogger << "Failed to find neighbours." << std::endl << write; 
	 continue;
      }
      if (Main::cellPtr != NULL) cpu_translation3(*Main::cellPtr,Main::nbrPtrs);
   }
   profile::stop("Compute inner cells",Main::cells.size(),"SpatialCells");
   computedCells=Main::cells.size();
   profile::start("Wait for receives");
   // Propagate boundary cells when transfers have completed:
   mpiGrid.wait_neighbour_data_update_receives();
   profile::stop("Wait for receives");
   profile::start("Compute border cells");
   
   Main::cells = mpiGrid.get_cells_with_remote_neighbour();
   for (size_t c=0; c<Main::cells.size(); ++c) {
      Main::cellPtr = mpiGrid[Main::cells[c]];
      if (findNeighbours(Main::nbrPtrs,mpiGrid,Main::cells[c]) == false) {
	 mpilogger << "Failed to find neighbours." << std::endl << write;
	 continue;
      }
      if (Main::cellPtr != NULL) cpu_translation3(*Main::cellPtr,Main::nbrPtrs);
   }
   profile::stop("Compute border cells",Main::cells.size(),"SpatialCells");
   computedCells+=Main::cells.size();
   profile::start("Wait for sends");
   mpiGrid.wait_neighbour_data_update_sends();
   profile::stop("Wait for sends");
   profile::stop("calcSpatProp",computedCells,"SpatialCells");
}



#endif // #ifndef PARGRID

