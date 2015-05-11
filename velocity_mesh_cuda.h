
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
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
//

/*todo
 -  make blocksize, gridlength cellsize static, and also the function computing stuff for them
*/

//#define CUDA_PERF_DEBUG      


namespace vmesh {
   template<typename GID,typename LID>
   class VelocityMeshCuda {
   public:
      __host__   void h_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3]);
      __device__ void d_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3]);
      __host__   void h_clear();
      __device__ void d_clear();
      __device__ __host__ void getIndices(const GID& globalID, LID& i,LID& j,LID& k);
      __device__ __host__ GID getGlobalID(LID indices[3]);
      __device__ __host__ LID size();
      __device__ __host__ LID numColumns();
      __device__ LID columnSize(LID column);
      
      Realf cellSize[3]; /**< Size (in m) of a cell in a block*/
      Realf blockSize[3]; /**< Size (in m) of a block*/
      LID gridLength[3];  /**< Max number of blocks per dim in block  grid*/
      uint nBlocks;
      uint nColumns;
      Realf *data;  
      GID *blockIDs;
      GID *blockIDsMapped; //temporary array for storing "rotated" blockIDs where the dimenionality is taken into account
      LID *sortedBlockLID;   //After sorting into columns, this tells the position (local ID) of a block in the unsorted data array
      LID *columnStartLID;   //The start LID of each column in sorted data.
   };
   
   

   
   template<typename GID, typename LID> __host__ void createVelocityMeshCuda(VelocityMeshCuda<GID,LID>** d_vmesh,
                                                                             VelocityMeshCuda<GID, LID>** h_vmesh,
                                                                             LID nBlocks,
                                                                             const LID gridLength[3],
                                                                             const Realf blockSize[3]);

   template<typename GID, typename LID> __host__ void uploadMeshData(VelocityMeshCuda<GID,LID>* d_vmesh,
                                                                     VelocityMeshCuda<GID, LID>* h_vmesh,
                                                                     const Realf *h_data,
                                                                     const GID *h_blockIDs,
                                                                     cudaStream_t stream);

   template<typename GID, typename LID> __host__ void destroyVelocityMeshCuda(VelocityMeshCuda<GID, LID> *d_vmesh,
                                                                              VelocityMeshCuda<GID, LID>* h_vmesh);
   
   template<typename GID, typename LID> __host__ void sortVelocityBlocksInColumns(VelocityMeshCuda<GID, LID> *d_vmesh,
                                                                                  VelocityMeshCuda<GID, LID>* h_vmesh,
                                                                                  uint dimension,
                                                                                  cudaStream_t stream);

   template<typename GID, typename LID> __host__ void createTargetMesh(VelocityMeshCuda<GID,LID>** d_targetVmesh, VelocityMeshCuda<GID,LID>** h_targetVmesh,
                                                                       VelocityMeshCuda<GID,LID>* d_sourceVmesh, VelocityMeshCuda<GID,LID>* h_sourceVmesh);
                                                                       

   
   template<typename GID, typename LID> __global__ void prepareSort(VelocityMeshCuda<GID,LID> *d_vmesh, uint dimension);
   template<typename GID, typename LID> __global__ void prepareColumnCompute(VelocityMeshCuda<GID,LID> *d_vmesh);

   
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
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::h_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3]){
      this->nBlocks=nBlocks;
      nColumns = 0; //not yet computed
      //cudaMalloc blocks...
      cudaMalloc(&data, nBlocks * WID3 * sizeof(Realf));
      cudaMalloc(&blockIDs, nBlocks * sizeof(GID));
      cudaMalloc(&blockIDsMapped, nBlocks * sizeof(GID));
      cudaMalloc(&sortedBlockLID, nBlocks * sizeof(LID));
      cudaMalloc(&columnStartLID, nBlocks * sizeof(LID));
      
      for(int i=0;i < 3; i++){
         this->gridLength[i] = gridLength[i];
         this->blockSize[i] = blockSize[i];
         this->cellSize[i] = blockSize[i] / WID;
      }
   }


   
   /*init on device side*/
   template<typename GID, typename LID> __device__ void VelocityMeshCuda<GID,LID>::d_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3]){
      this->nBlocks=nBlocks;
      nColumns = 0; //not yet computed
      data = (Realf*) malloc(nBlocks * WID3 * sizeof(Realf));
      blockIDs = (GID*) malloc(nBlocks * sizeof(GID));
      blockIDsMapped = (GID*) malloc(nBlocks * sizeof(GID));
      sortedBlockLID = (LID*) malloc(nBlocks * sizeof(LID));
      columnStartLID = (LID*) malloc(nBlocks * sizeof(LID));
      
      for(int i=0;i < 3; i++){
         this->gridLength[i] = gridLength[i];
         this->blockSize[i] = blockSize[i];
         this->cellSize[i] = blockSize[i] / WID;
      }
   }

   
   /*free on host side*/
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::h_clear(){
      nBlocks = 0;
      nColumns = 0;
      cudaFree(data);
      cudaFree(blockIDs);
      cudaFree(blockIDsMapped);
      cudaFree(sortedBlockLID);
      cudaFree(columnStartLID);
   }

   
   /*free on host side*/
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::d_clear(){
      nBlocks = 0;
      nColumns = 0;
      free(data);
      free(blockIDs);
      free(blockIDsMapped);
      free(sortedBlockLID);
      free(columnStartLID);
   }

   template<typename GID, typename LID> __device__ __host__ LID VelocityMeshCuda<GID,LID>::size(){
         return nBlocks;
   }

   template<typename GID, typename LID> __device__ __host__ LID VelocityMeshCuda<GID,LID>::numColumns(){
         return nColumns;
   }
   
   template<typename GID, typename LID> __device__ LID VelocityMeshCuda<GID,LID>::columnSize(LID column){
      return ((column < nColumns - 1) ? columnStartLID[column + 1] : nBlocks ) - columnStartLID[column];
   }
   
/*---------------------------------------- INTERFACE functions ------------------------------------------*/

   
   template<typename GID,typename LID>
   __host__ void createVelocityMeshCuda(VelocityMeshCuda<GID,LID>** d_vmesh, VelocityMeshCuda<GID,LID>** h_vmesh, 
                                        LID nBlocks, const LID gridLength[3], const Realf blockSize[3]){
      //allocate space on device for device resident class
      cudaMalloc(d_vmesh, sizeof(VelocityMeshCuda<GID, LID>));      
      cudaMallocHost(h_vmesh, sizeof(VelocityMeshCuda<GID, LID>));
      
      //init members on host
      (*h_vmesh)->h_init(nBlocks, gridLength, blockSize);
      //copy all  members to device
      cudaMemcpy((*d_vmesh), (*h_vmesh), sizeof(VelocityMeshCuda<GID, LID>), cudaMemcpyHostToDevice);

   }
/*
   template<typename GID, typename LID> __global__ void computeTargetGridColumns(VelocityMeshCuda<GID,LID> *d_vmesh, LID *targetColumns){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      if (id < d_vmesh->nBlocks ){
      }
   }
         
   template<typename GID,typename LID>
   __host__ void createTargetMesh(VelocityMeshCuda<GID,LID>** d_targetVmesh, VelocityMeshCuda<GID,LID>** h_targetVmesh,
                                  VelocityMeshCuda<GID,LID>* d_sourceVmesh,  VelocityMeshCuda<GID,LID>*  h_sourceVmesh,
                                  cudaStream_t stream) {
      //compute target  grid
      
      

   }

*/ 

   
   
   template<typename GID,typename LID> __host__ void uploadMeshData(VelocityMeshCuda<GID,LID>* d_vmesh, VelocityMeshCuda<GID,LID>* h_vmesh, 
                                                                    const Realf *h_data, const  GID *h_blockIDs, cudaStream_t stream) {
      cudaMemcpyAsync(h_vmesh->data, h_data, h_vmesh->size() *  WID3 * sizeof(Realf), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(h_vmesh->blockIDs, h_blockIDs, h_vmesh->size() * sizeof(GID), cudaMemcpyHostToDevice, stream);
   }

   template<typename GID, typename LID> __global__ void prepareSort(VelocityMeshCuda<GID,LID> *d_vmesh, uint dimension){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      if (id < d_vmesh->nBlocks ){
         LID blocki = id;
         GID block = d_vmesh->blockIDs[blocki];
         d_vmesh->sortedBlockLID[blocki] = blocki; //reset offsets
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
                const LID x_indice = block % d_vmesh->gridLength[0];
                const LID y_indice = (block / d_vmesh->gridLength[0]) % d_vmesh->gridLength[1];
                // Mapping the block id to different coordinate system if dimension is not zero:
                d_vmesh->blockIDsMapped[blocki] = block - (x_indice + y_indice*d_vmesh->gridLength[0]) + y_indice + x_indice * d_vmesh->gridLength[1];
                
             }
                break;
             case 2: {
                // Do operation: 
                //   block = x + y*x_max + z*y_max*x_max 
                //=> block' = z + y*z_max + x*z_max*y_max
                const LID x_indice = block % d_vmesh->gridLength[0];
                const LID y_indice = (block / d_vmesh->gridLength[0]) % d_vmesh->gridLength[1];
                const LID z_indice =  (block / (d_vmesh->gridLength[0] * d_vmesh->gridLength[1]));
                // Mapping the block id to different coordinate system if dimension is not zero:
                d_vmesh->blockIDsMapped[blocki] =  z_indice + y_indice * d_vmesh->gridLength[2] + x_indice * d_vmesh->gridLength[1] * d_vmesh->gridLength[2];
             }
                break;
         }
      }
   }
   
   template<typename GID, typename LID> __global__ void prepareColumnCompute(VelocityMeshCuda<GID,LID> *d_vmesh){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      if (id < d_vmesh->nBlocks ){
         //check if the mapped (so blockids in the dimension we are computing) is contiguous or not.
         if(id > 0 && d_vmesh->blockIDsMapped[id-1] != d_vmesh->blockIDsMapped[id] - 1 )
            d_vmesh->columnStartLID[id] = id;
         else
            d_vmesh->columnStartLID[id] = 0;
      }
      //            }
   }


   struct isZero{
      __host__ __device__ bool operator()(const uint x) {
         return x == 0;
      }
   };
   
      
   //assume h_vmesh and d_vmesh are synchronized
   template<typename GID, typename LID> __host__ void sortVelocityBlocksInColumns(VelocityMeshCuda<GID, LID> *d_vmesh, VelocityMeshCuda<GID,LID> *h_vmesh, uint dimension, cudaStream_t stream) {
      int cuBlockSize = 512; 
      int cuGridSize = 1 + h_vmesh->size() / cuBlockSize; // value determine by block size and total work
      vmesh::prepareSort<<<cuGridSize, cuBlockSize, 0, stream>>>(d_vmesh, dimension);      
      thrust::sort_by_key(thrust::cuda::par.on(stream),
                          h_vmesh->blockIDsMapped, h_vmesh->blockIDsMapped + h_vmesh->size(),
                          h_vmesh->sortedBlockLID);
      vmesh::prepareColumnCompute<<<cuGridSize, cuBlockSize, 0, stream>>>(d_vmesh);
      LID *newEnd = thrust::remove_if(thrust::cuda::par.on(stream),
                                      h_vmesh->columnStartLID + 1,
                                      h_vmesh->columnStartLID + h_vmesh->size(),
                                      isZero()
                                      );
      h_vmesh->nColumns = newEnd - h_vmesh->columnStartLID;       
      //copy nColumns to device version
      cudaMemcpy(d_vmesh, h_vmesh, sizeof(VelocityMeshCuda<GID, LID>), cudaMemcpyHostToDevice);      


      
      const bool debugSort=true;
      if(debugSort) {
         //print out result of sort
         GID *blockIDs = (GID *) malloc(sizeof(GID) *  h_vmesh->size());
         GID *blockIDsMapped = (GID *) malloc(sizeof(GID) *  h_vmesh->size());
         LID *sortedBlockLID = (LID *) malloc(sizeof(LID) * h_vmesh->size());
         LID *columnStartLID = (LID *) malloc(sizeof(LID) * h_vmesh->size());
         
         cudaMemcpyAsync(blockIDs, h_vmesh->blockIDs , sizeof(GID) * h_vmesh->size() , cudaMemcpyDeviceToHost, stream);
         cudaMemcpyAsync(blockIDsMapped, h_vmesh->blockIDsMapped , sizeof(GID) * h_vmesh->size() , cudaMemcpyDeviceToHost, stream);
         cudaMemcpyAsync(sortedBlockLID, h_vmesh->sortedBlockLID , sizeof(LID) * h_vmesh->size() , cudaMemcpyDeviceToHost, stream);
         cudaMemcpyAsync(columnStartLID, h_vmesh->columnStartLID , sizeof(LID) * h_vmesh->size() , cudaMemcpyDeviceToHost, stream);
      
         cudaStreamSynchronize(stream);

         printf("--------------------------------------------\n");
         printf("Sort results - dimension = %d\n", dimension);
         printf("Host gridLength =  %d %d %d\n", h_vmesh->gridLength[0], h_vmesh->gridLength[1], h_vmesh->gridLength[2]);
         printf("mappedblock[i] block[offset[i]] blockx,y,z\n");
         LID c = 0;
         for(LID i = 0; i < h_vmesh->size(); i++){      
            if(columnStartLID[c] == i) {
               printf("--- Column %d / %d  start %d --- \n", c, h_vmesh->nColumns, columnStartLID[c]);
               c++;
            }
            LID x,y,z;
            h_vmesh->getIndices(blockIDs[sortedBlockLID[i]], x, y, z);
            printf("%d %d %d,%d,%d\n", blockIDsMapped[i], blockIDs[sortedBlockLID[i]], x, y, z);
         }            
         printf("--------------------------------------------\n");

         free(blockIDs);
         free(blockIDsMapped);
         free(sortedBlockLID);
      }
   }
   
   template<typename GID, typename LID> __host__ void destroyVelocityMeshCuda(VelocityMeshCuda<GID,LID> *d_vmesh,VelocityMeshCuda<GID,LID> *h_vmesh) {
      //copy all  members to host (not deep, so has pointers to device arrays)
      cudaMemcpy(h_vmesh, d_vmesh, sizeof(VelocityMeshCuda<GID,LID>), cudaMemcpyDeviceToHost);
      //De-allocate cuda arrays in velocity mesh
      h_vmesh->h_clear();
      //free also the mesh itself
      cudaFree(d_vmesh);
      cudaFreeHost(h_vmesh);

      //h_vmesh will now be deallocated
   }
   
}; // namespace vmesh

#endif
