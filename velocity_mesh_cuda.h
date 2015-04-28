
/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2014 Finnish Meteorological Institute
 */

#ifndef VELOCITY_MESH_CUDA_H
#define VELOCITY_MESH_CUDA_H

#include "common.h" //this gives GlobalID and LocalID types
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#warning "Integrate this Realf, Real with the vec.h machinery"
#define Realf float
#define Real double

#else
#define HOST_DEVICE
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

//

/*todo

 -  make blocksize, gridlength cellsize static, and also the function computing stuff for them


*/

namespace vmesh {
   template<typename GID,typename LID>
   class VelocityMeshCuda {
   public:
      __host__ void init(const Realf *h_data, const GID *h_blockIDs, uint nBlocks,
                         const LID gridLength[3], const Realf blockSize[3],
                         cudaStream_t stream);
      __host__ void clear();
      __device__ __host__ void getIndices(const GID& globalID, LID& i,LID& j,LID& k);
      __device__ __host__ GID  getGlobalID(LID indices[3]);
      __device__ __host__ uint size();
      
      Realf cellSize[3]; /**< Size (in m) of a cell in a block*/
      Realf blockSize[3]; /**< Size (in m) of a block*/
      LID gridLength[3];  /**< Max number of blocks per dim in block  grid*/
      uint nBlocks;
      Realf *data;
      GID *blockIDs;
      GID *blockIDsMapped; //temporary array for storing "rotated" blockIDs where the dimenionality is taken into account
      LID *blockOffset;
   };



   
   template<typename GID, typename LID> __host__ VelocityMeshCuda<GID,LID>* createVelocityMeshCuda(const Realf *h_data, const GID *h_blockIDs, LID nBlocks,
                                                                                                   const LID gridLength[3], const Realf blockSize[3],
                                                                                                   cudaStream_t stream);
   template<typename GID, typename LID> __host__ void destroyVelocityMeshCuda(VelocityMeshCuda<GID, LID> *d_vmesh, cudaStream_t stream);
   template<typename GID, typename LID> __host__ void sortVelocityBlocks(VelocityMeshCuda<GID, LID> *d_vmesh, uint dimension, cudaStream_t stream);
   
   template<typename GID, typename LID> __global__ void initBlockOffsets(VelocityMeshCuda<GID,LID> *d_vmesh);
   template<typename GID, typename LID> __global__ void testBlockOffsets(VelocityMeshCuda<GID,LID> *d_vmesh, int testId);
   template<typename GID, typename LID> __global__ void prepareSort(VelocityMeshCuda<GID,LID> *d_vmesh, uint dimension);
   
   
/*----------------------------------------CLASS functions ------------------------------------------*/



   template<typename GID,typename LID> inline
   void VelocityMeshCuda<GID,LID>::getIndices(const GID& globalID, LID& i,LID& j,LID& k) {
      if (globalID >= INVALID_GLOBALID) {
         i = j = k = INVALID_LOCALID;
      } else {
         i = globalID % gridLength[0];
         j = (globalID / gridLength[0]) % gridLength[1];
         k = globalID / (gridLength[0] * gridLength[1]);
      }
   }

      
   template<typename GID,typename LID> inline
   GID VelocityMeshCuda<GID,LID>::getGlobalID(LID indices[3]) {
      if (indices[0] >= gridLength[0]) return INVALID_GLOBALID;
      if (indices[1] >= gridLength[1]) return INVALID_GLOBALID;
      if (indices[2] >= gridLength[2]) return INVALID_GLOBALID;
      return indices[2]*gridLength[1]*gridLength[0] + indices[1]*gridLength[0] + indices[0];
   }
 
   
   /*init on host side*/
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::init(const Realf *h_data, const GID *h_blockIDs, uint h_nBlocks,
                                                                                      const LID gridLength[3], const Realf blockSize[3],
                                                                                      cudaStream_t stream) {
      cudaEvent_t evA, evB;
      cudaEventCreate(&evA);
      cudaEventCreate(&evB);
      
      nBlocks=h_nBlocks;
      cudaMalloc(&data, nBlocks * WID3 * sizeof(Realf));
      cudaMalloc(&blockIDs, nBlocks * sizeof(GID));
      cudaMalloc(&blockIDsMapped, nBlocks * sizeof(GID));
      cudaMalloc(&blockOffset, nBlocks * sizeof(LID));
      

      for(int i=0;i < 3; i++){
         this->gridLength[i] = gridLength[i];
         this->blockSize[i] = blockSize[i];
         this->cellSize[i] = blockSize[i] / WID;
      }
      
      uint bytes = nBlocks *  WID3 * sizeof(Realf) +nBlocks * sizeof(GID);
      float milliseconds=0;

      cudaEventRecord(evA, stream);      
      cudaMemcpyAsync(data, h_data, nBlocks *  WID3 * sizeof(Realf), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(blockIDs, h_blockIDs, nBlocks * sizeof(GID), cudaMemcpyHostToDevice, stream);
      cudaEventRecord(evB, stream);
      cudaEventSynchronize(evB);
      cudaEventElapsedTime(&milliseconds, evA, evB);  
      printf("cudaMemcpyAsync transferred %d blocks,  %d bytes in %g s to GPU: %g GB/s. \n", nBlocks, bytes, milliseconds * 1e-3, (bytes * 1e-9) / (milliseconds * 1e-3) );
   }

   /*free on host side*/
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::clear(){
      cudaFree(data);
      cudaFree(blockIDs);
      cudaFree(blockIDsMapped);
      cudaFree(blockOffset);
   }
   
   template<typename GID, typename LID> __device__ __host__ uint VelocityMeshCuda<GID,LID>::size(){
      return nBlocks;
   }
   
/*---------------------------------------- INTERFACE functions ------------------------------------------*/

   
   
   template<typename GID,typename LID>
   __host__ VelocityMeshCuda<GID,LID>* createVelocityMeshCuda(const Realf *h_data, const  GID *h_blockIDs, LID nBlocks,
                                                              const LID gridLength[3], const Realf blockSize[3],
                                                              cudaStream_t stream) {
      int cuBlockSize = 512; 
      int cuGridSize = 1 + nBlocks / cuBlockSize; // value determine by block size and total work
      VelocityMeshCuda<GID,LID> h_vmesh;
      VelocityMeshCuda<GID,LID> *d_vmesh;
      //allocate space on device for device resident class
      cudaMalloc((void **)&d_vmesh, sizeof(VelocityMeshCuda<GID, LID>));      
      //init members on host
      h_vmesh.init(h_data, h_blockIDs, nBlocks,
                   gridLength, blockSize,
                   stream);
      //copy all  members to device
      cudaStreamSynchronize(stream);
      cudaMemcpyAsync(d_vmesh, &h_vmesh, sizeof(VelocityMeshCuda<GID, LID>), cudaMemcpyHostToDevice, stream);

      cudaStreamSynchronize(stream);
      vmesh::initBlockOffsets<<<cuGridSize, cuBlockSize, 0, stream>>>(d_vmesh);
      
      cudaStreamSynchronize(stream);
      vmesh::testBlockOffsets<<<cuGridSize, cuBlockSize, 0, stream>>>(d_vmesh, 610);

      return d_vmesh;
       //h_vmesh will now be deallocated
   }


   template<typename GID, typename LID> __global__ void initBlockOffsets(VelocityMeshCuda<GID,LID> *d_vmesh){
      int id = blockIdx.x*blockDim.x+threadIdx.x;
      if (id < d_vmesh->nBlocks ){
         d_vmesh->blockOffset[id] = id;
      }
   }


   template<typename GID, typename LID> __global__ void prepareSort(VelocityMeshCuda<GID,LID> *d_vmesh, uint dimension){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      if (id < d_vmesh->nBlocks ){
         const LID blocki = id; // blockOffset[id] not needed here, keeps memory access coalesced
         const GID block = d_vmesh->blockIDs[blocki];
         switch( dimension ) {
             case 0: {
                d_vmesh->blockIDsMapped[blocki] = block; // Mapping the block id to different coordinate system if dimension is not zero:
             }
                break;
             case 1: {
                // Do operation: 
                //   block = x + y*x_max + z*y_max*x_max 
                //=> block' = block - (x + y*x_max) + y + x*y_max = x + y*x_max + z*y_max*x_max - (x + y*x_max) + y + x*y_max
                //          = y + x*y_max + z*y_max*x_max
                const uint x_indice = block%d_vmesh->gridLength[0];
                const uint y_indice = (block/d_vmesh->gridLength[0])%d_vmesh->gridLength[1];
                // Mapping the block id to different coordinate system if dimension is not zero:
                d_vmesh->blockIDsMapped[blocki] = block - (x_indice + y_indice*d_vmesh->gridLength[0]) + y_indice + x_indice * d_vmesh->gridLength[1];
                
             }
                break;
             case 2: {
                // Do operation: 
                //   block = x + y*x_max + z*y_max*x_max 
                //=> block' = z + y*z_max + x*z_max*y_max
                const uint x_indice = block%d_vmesh->gridLength[0];
                const uint y_indice = (block/d_vmesh->gridLength[0])%d_vmesh->gridLength[1];
                const uint z_indice =  (block/(d_vmesh->gridLength[0]*d_vmesh->gridLength[1]));
                // Mapping the block id to different coordinate system if dimension is not zero:
                d_vmesh->blockIDsMapped[blocki] = z_indice + y_indice * d_vmesh->gridLength[2] + x_indice*d_vmesh->gridLength[1]*d_vmesh->gridLength[2];
             }
                break;
         }
      }
   }
   
   template<typename GID, typename LID> __global__ void testBlockOffsets(VelocityMeshCuda<GID,LID> *d_vmesh, int testId) {
      int id = blockIdx.x*blockDim.x+threadIdx.x;
      if( testId == id && id < d_vmesh->nBlocks ){
         printf("id %d (block %d, thread %d) has a value of %d\n",id, blockIdx.x, threadIdx.x, d_vmesh->blockOffset[id]);
      }
   }
   template<typename GID, typename LID> __host__ void sortVelocityBlocks(VelocityMeshCuda<GID, LID> *d_vmesh, uint dimension, cudaStream_t stream) {
      int cuBlockSize = 512; 
      int cuGridSize = 1 + d_vmesh->size() / cuBlockSize; // value determine by block size and total work
      vmesh::prepareSort<<<cuGridSize, cuBlockSize, 0, stream>>>(d_vmesh);
//      thrust::sort(thrust::cuda::par.on(s), keys.begin(), keys.end());

   }
   
   template<typename GID, typename LID> __host__ void destroyVelocityMeshCuda(VelocityMeshCuda<GID,LID> *d_vmesh, cudaStream_t stream) {
      VelocityMeshCuda<GID,LID> h_vmesh;
      //copy all  members to host (not deep), not async since we need to 
      cudaMemcpyAsync(&h_vmesh, d_vmesh, sizeof(VelocityMeshCuda<GID,LID>), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      //clear data in velocity mesh
      h_vmesh.clear();
      //free also the mesh itself
      cudaFree(d_vmesh);
      //h_vmesh will now be deallocated
   }
   
}; // namespace vmesh

#endif
