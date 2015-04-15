/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2014 Finnish Meteorological Institute
 */

#ifndef VELOCITY_MESH_CUDA_H
#define VELOCITY_MESH_CUDA_H

#include "common.h"
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#warning "Integrate this Realf with the vec.h machinery"
#define Realf float
#else
#define HOST_DEVICE
#endif

//

namespace vmesh {
   class VelocityMeshCuda {
   public:
      __device__ VelocityMeshCuda();
      __device__ ~VelocityMeshCuda();
      
   private:
      Realf dv;
      int columnDimension;
      Realf *data;
      int3 *columnBeginCoordinate; //x,y,z coordinate where column starts
      int *columnOffset; //Offset in data where the column starts
      int *nz;    //Length of column in z direction, size is nx*ny
      int *ny;    //Length of column in z direction, size is nx*ny

      
   };
   template<typename GID> __host__ void  readInMesh(Realf *h_data, GID *h_blockIDs, uint nBlocks) {

      Realf *d_data;
      GID *d_blockIDs;
      VelocityMeshCuda *vmeshes;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      
      cudaMalloc(&d_data, nBlocks * WID3 * sizeof(Realf));
      cudaMalloc(&d_blockIDs, nBlocks * sizeof(GID));
      cudaEventRecord(start);
      cudaMemcpy(d_data, h_data, nBlocks *  WID3 * sizeof(Realf), cudaMemcpyHostToDevice);
      cudaMemcpy(d_blockIDs, h_blockIDs, nBlocks * sizeof(GID), cudaMemcpyHostToDevice);
      cudaEventRecord(stop);
      
      cudaEventSynchronize(stop);
      uint bytes = nBlocks *  WID3 * sizeof(Realf) +nBlocks * sizeof(GID);
      float milliseconds=0;
      cudaEventElapsedTime(&milliseconds, start, stop);  
      printf("Transferred %d blocks,  %d bytes in %g s to GPU: %g GB/s. \n", nBlocks, bytes, milliseconds * 1e-3, (bytes * 1e-9) / (milliseconds * 1e-3) );
   
   }


}; // namespace vmesh

#endif
