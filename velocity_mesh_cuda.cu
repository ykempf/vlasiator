/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2014 Finnish Meteorological Institute
 */
#include "velocity_mesh_cuda.h"


//using namespace vmesh;


__device__  vmesh::VelocityMeshCuda::VelocityMeshCuda() { }


__device__  vmesh::VelocityMeshCuda::~VelocityMeshCuda() { }




__global__ void vmesh::readInMesh(realf *d_data, GlobalID *d_blockIDs, uint nBlocks){


}




