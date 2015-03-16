/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2014 Finnish Meteorological Institute
 */

#ifndef VELOCITY_MESH_CUDA_H
#define VELOCITY_MESH_CUDA_H

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif


namespace vmesh {

   class VelocityMeshCuda {
    public:      
      __host__ __device__ VelocityMeshCuda();
      __host__ __device__ ~VelocityMeshCuda();
      //      __host__ uploadVelocityMesh(SpatialCell* cell);
      //     __host__ downloadVelocityMesh(SpatialCell* cell);
      
   private:
      float dv;
      int columnDimension;
      float *data;
      int3 *columnBeginCoordinate; //x,y,z coordinate where column starts
      int *columnOffset; //Offset in data where the column starts
      int *nz;    //Length of column in z direction, size is nx*ny
      int *ny;    //Length of column in z direction, size is nx*ny
      
};

   
}; // namespace vmesh

#endif
