
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
 -  make blocksize, gridlength cellsize static, and also the function computing stuff for them, 
*/

//#define CUDA_PERF_DEBUG      


namespace vmesh {
   template<typename GID,typename LID>
   class VelocityMeshCuda {
   public:
      __host__   void h_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3], const Realf gridMinLimits[3]);
      __device__ void d_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3], const Realf gridMinLimits[3]);
      __host__   void h_clear();
      __device__ void d_clear();
      __device__ __host__ void getIndices(const GID& globalID, LID blockIndices[3]);
      __device__ __host__ void transposeIndices(LID blockIndices[3], uint dimension);
      __device__ __host__ GID getGlobalID(LID blockIndices[3]);
      __device__ __host__ LID size();
      __device__ __host__ LID numColumns();
      __device__ LID columnSize(LID column);
      
      Realf cellSize[3];      /**< Size (in m) of a cell in a block*/
      Realf blockSize[3];     /**< Size (in m) of a block*/
      Realf gridMinLimits[3]; /**< Minimum coordinate (m) of the grid */
      LID gridLength[3];      /**< Max number of blocks per dim in block  grid*/
      uint nBlocks;
      uint nColumns;
      uint sortDimension;    /**< dimension (0,1,2) along which blocks are sorted*/
      Realf *data;  
      GID *blockIDs;
      GID *sortedBlockMappedGID;   /**< Array for storing rotated/mapped blockIDs where the dimenionality is taken into account */
      LID *sortedBlockLID;         /**< After sorting into columns, this tells the position ( local ID) of a block in the unsorted data array */
      LID *columnStartLID;         /**< The start LID of each column in sorted data.*/
   };
   
   

   /*external interface */
   template<typename GID, typename LID> __host__ void createVelocityMeshCuda(VelocityMeshCuda<GID, LID>** d_vmesh,
                                                                             VelocityMeshCuda<GID, LID>** h_vmesh,
                                                                             LID nBlocks,
                                                                             const LID gridLength[3],
                                                                             const Realf blockSize[3],
                                                                             const Realf gridMinLimits[3]
                                                                             );
   template<typename GID, typename LID> __host__ void destroyVelocityMeshCuda(VelocityMeshCuda<GID, LID> *d_vmesh,
                                                                              VelocityMeshCuda<GID, LID>* h_vmesh);

   
   template<typename GID, typename LID> __host__ void uploadMeshData(VelocityMeshCuda<GID,LID>* d_vmesh,
                                                                     VelocityMeshCuda<GID, LID>* h_vmesh,
                                                                     const Realf *h_data,
                                                                     const GID *h_blockIDs,
                                                                     cudaStream_t stream);

   template<typename GID, typename LID> __host__ void sortVelocityBlocksInColumns(VelocityMeshCuda<GID, LID> *d_vmesh,
                                                                                  VelocityMeshCuda<GID, LID>* h_vmesh,
                                                                                  uint dimension,
                                                                                  cudaStream_t stream);
   
   template<typename GID, typename LID> __host__ void createTargetMesh(VelocityMeshCuda<GID,LID>** d_targetVmesh, VelocityMeshCuda<GID,LID>** h_targetVmesh,
                                                                       VelocityMeshCuda<GID,LID>* d_sourceVmesh, VelocityMeshCuda<GID,LID>* h_sourceVmesh,
                                                                       Real* intersections,
                                                                       uint dimension, cudaStream_t stream);

   /*internal functions and kernels*/


   template<typename GID, typename LID> __global__ void prepareSort(VelocityMeshCuda<GID,LID> *d_vmesh, uint dimension);
   template<typename GID, typename LID> __global__ void prepareColumnCompute(VelocityMeshCuda<GID,LID> *d_vmesh);

   
/*----------------------------------------CLASS functions ------------------------------------------*/

   
   
   template<typename GID,typename LID> inline
   void VelocityMeshCuda<GID,LID>::getIndices(const GID& globalID, LID blockIndices[3]) {
      if (globalID >= INVALID_GLOBALID) {
         blockIndices[0] = blockIndices[1] = blockIndices[2] = INVALID_LOCALID;
      } else {
         blockIndices[0] = globalID % gridLength[0];
         blockIndices[1] = (globalID / gridLength[0]) % gridLength[1];
         blockIndices[2] = globalID / (gridLength[0] * gridLength[1]);
      }
   }
   

   template<typename GID,typename LID> inline
   void VelocityMeshCuda<GID,LID>::transposeIndices(LID blockIndices[3], uint dimension) {
      //Switch block indices according to dimensions, the slice3d algorithm has
      //  been written for integrating along z.
      LID temp;
      switch (dimension){
          case 0:
             /*i and k coordinates have been swapped*/
             temp=blockIndices[2];
             blockIndices[2]=blockIndices[0];
             blockIndices[0]=temp;
             break;
          case 1:
             /*in values j and k coordinates have been swapped*/
             temp=blockIndices[2];
             blockIndices[2]=blockIndices[1];
             blockIndices[1]=temp;
             break;
          case 2:
             break;
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
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::h_init(uint nBlocks, const LID gridLength[3], const Realf blockSize[3], const Realf gridMinLimits[3]){
      this->nBlocks=nBlocks;
      nColumns = 0; //not yet computed
      //cudaMalloc blocks...
      cudaMalloc(&data, nBlocks * WID3 * sizeof(Realf));
      cudaMalloc(&blockIDs, nBlocks * sizeof(GID));
      cudaMalloc(&sortedBlockMappedGID, nBlocks * sizeof(GID));
      cudaMalloc(&sortedBlockLID, nBlocks * sizeof(LID));
      cudaMalloc(&columnStartLID, nBlocks * sizeof(LID));
      
      for(int i=0;i < 3; i++){
         this->gridLength[i] = gridLength[i];
         this->blockSize[i] = blockSize[i];
         this->cellSize[i] = blockSize[i] / WID;
         this->gridMinLimits[i] = gridMinLimits[i];
      }
   }


   
   /*init on device side*/
   template<typename GID, typename LID> 
   __device__ void VelocityMeshCuda<GID,LID>::d_init(uint nBlocks, 
                                                     const LID gridLength[3], 
                                                     const Realf blockSize[3], 
                                                     const Realf gridMinLimits[3]){
      this->nBlocks=nBlocks;
      nColumns = 0; //not yet computed
      data = (Realf*) malloc(nBlocks * WID3 * sizeof(Realf));
      blockIDs = (GID*) malloc(nBlocks * sizeof(GID));
      sortedBlockMappedGID = (GID*) malloc(nBlocks * sizeof(GID));
      sortedBlockLID = (LID*) malloc(nBlocks * sizeof(LID));
      columnStartLID = (LID*) malloc(nBlocks * sizeof(LID));
      
      for(int i=0;i < 3; i++){
         this->gridLength[i] = gridLength[i];
         this->blockSize[i] = blockSize[i];
         this->cellSize[i] = blockSize[i] / WID;
         this->gridMinLimits[i] = gridMinLimits[i];
      }
   }

   
   /*free on host side*/
   template<typename GID, typename LID> __host__ void VelocityMeshCuda<GID,LID>::h_clear(){
      nBlocks = 0;
      nColumns = 0;
      cudaFree(data);
      cudaFree(blockIDs);
      cudaFree(sortedBlockMappedGID);
      cudaFree(sortedBlockLID);
      cudaFree(columnStartLID);
   }

   
   /*free on host side*/
   template<typename GID, typename LID> __device__ void VelocityMeshCuda<GID,LID>::d_clear(){
      nBlocks = 0;
      nColumns = 0;
      free(data);
      free(blockIDs);
      free(sortedBlockMappedGID);
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
    __host__ void createVelocityMeshCuda(VelocityMeshCuda<GID,LID>** d_vmesh, 
                                         VelocityMeshCuda<GID,LID>** h_vmesh, 
                                         LID nBlocks, 
                                         const LID gridLength[3], 
                                         const Realf blockSize[3], 
                                         const Realf gridMinLimits[3]){
      //allocate space on device for device resident class
      cudaMalloc(d_vmesh, sizeof(VelocityMeshCuda<GID, LID>));      
      cudaMallocHost(h_vmesh, sizeof(VelocityMeshCuda<GID, LID>));
      
      //init members on host
      (*h_vmesh)->h_init(nBlocks, gridLength, blockSize, gridMinLimits);
      //copy all  members to device
      cudaMemcpy((*d_vmesh), (*h_vmesh), sizeof(VelocityMeshCuda<GID, LID>), cudaMemcpyHostToDevice);

   }

   template<typename GID, typename LID> __global__ void computeTargetGridColumns(VelocityMeshCuda<GID,LID> *d_vmesh,
                                                                                 LID *targetColumnLength,
                                                                                 GID *targetColumnFirstBlock,
                                                                                 Real intersection,
                                                                                 Real intersection_di,
                                                                                 Real intersection_dj,
                                                                                 Real intersection_dk,
                                                                                 uint dimension){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      if (id < d_vmesh->nColumns){
         GID sourceStartBlock = d_vmesh->sortedBlockMappedGID[d_vmesh->columnStartLID[id]];
         GID sourceEndBlock = d_vmesh->sortedBlockMappedGID[d_vmesh->columnStartLID[id] + d_vmesh->columnSize(id)];
         LID indicesStart[3];
         LID indicesEnd[3];

         //get indices for first and last block in column
         d_vmesh->getIndices(sourceStartBlock, indicesStart);
         d_vmesh->getIndices(sourceEndBlock, indicesEnd );
         //Transpose indices, now everything is int terms of
         //transposed coordinate system where z is now along dimension
         //(also intersections should now be in this system)
         d_vmesh->transposeIndices(indicesStart, dimension );
         d_vmesh->transposeIndices(indicesEnd, dimension );
         
         Realf zMin, zMax;
         
//compute minimum z value of any cell in the block (factor of two already included in the intersections)
         zMin = intersection +
                min( (indicesStart[0] * WID + 0) * intersection_di, (indicesStart[0] * WID + WID-1) * intersection_di) +
                min( (indicesStart[1] * WID + 0) * intersection_dj, (indicesStart[1] * WID + WID-1) * intersection_dj) + 
                (indicesStart[2] * WID + 0) * intersection_dk;

         zMax = intersection +
                max( (indicesEnd[0] * WID + 0) * intersection_di, (indicesEnd[0] * WID + WID-1) * intersection_di) + 
                max( (indicesEnd[1] * WID + 0) * intersection_dj, (indicesEnd[1] * WID + WID-1) * intersection_dj) +
                (indicesEnd[2] * WID + WID) * intersection_dk;  
         
         printf("Column %d nblocks %d zmin %g zmax %g intersections %g %g %g %g\n", id, d_vmesh->nColumns, zMin, zMax, intersection, intersection_di, intersection_dj, intersection_dk );

         
         indicesStart[2]  = (int)((zMin-intersection)/intersection_dk);
         indicesEnd[2] = ((zMax-intersection)/intersection_dk);
         
         targetColumnLength[id] = indicesEnd[2] - indicesStart[2] + 1;

         //Transpose indices back to original coordinate system
         d_vmesh->transposeIndices(indicesStart, dimension );
         d_vmesh->transposeIndices(indicesEnd, dimension );
         
         targetColumnFirstBlock[id] = d_vmesh->getGlobalID(indicesStart);
         
         

      }   
         
   }  
   



   template<typename GID,typename LID>
   __host__ void createTargetMesh(VelocityMeshCuda<GID,LID>** d_targetVmesh, VelocityMeshCuda<GID,LID>** h_targetVmesh,
                                  VelocityMeshCuda<GID,LID>* d_sourceVmesh,  VelocityMeshCuda<GID,LID>*  h_sourceVmesh,
                                  Real intersection,
                                  Real intersection_di,
                                  Real intersection_dj,
                                  Real intersection_dk, 
                                  uint dimension, cudaStream_t stream) {
      int cuBlockSize = 512; 
      int cuGridSize = 1 + h_sourceVmesh->nColumns / cuBlockSize; // value determine by block size and total work   
      LID targetnBlocks;
       //TODO compute targetnBlocks  
      LID *targetColumnLengths;
      GID *targetColumnFirstBlock;
      cudaMalloc(&targetColumnLengths, sizeof(LID) * h_sourceVmesh->nColumns);
      cudaMalloc(&targetColumnFirstBlock, sizeof(GID) * h_sourceVmesh->nColumns); 
   
      vmesh::computeTargetGridColumns<<<cuGridSize, cuBlockSize, 0, stream>>>(d_sourceVmesh, 
                                                                              targetColumnLengths,
                                                                              targetColumnFirstBlock,
                                                                              intersection,
                                                                              intersection_di,
                                                                              intersection_dj,
                                                                              intersection_dk,
                                                                              dimension); 
      
 

//      createVelocityMeshCuda(d_targetVmesh, h_targetVmesh, targetnBlocks, 
  //                             d_sourceVmesh->gridLength, d_sourceVmesh->blockSize, d_sourceVmesh->gridMinLimits);
           
      cudaFree(targetColumnLengths);
      cudaFree(targetColumnFirstBlock);   
   }



   
   
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
                d_vmesh->sortedBlockMappedGID[blocki] = block; // Mapping the block id to different coordinate system if dimension is not zero:
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
                d_vmesh->sortedBlockMappedGID[blocki] = block - (x_indice + y_indice*d_vmesh->gridLength[0]) + y_indice + x_indice * d_vmesh->gridLength[1];
                
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
                d_vmesh->sortedBlockMappedGID[blocki] =  z_indice + y_indice * d_vmesh->gridLength[2] + x_indice * d_vmesh->gridLength[1] * d_vmesh->gridLength[2];
             }
                break;
         }
      }
   }

   template<typename GID, typename LID> __global__ void prepareColumnCompute(VelocityMeshCuda<GID,LID> *d_vmesh){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      if (id < d_vmesh->nBlocks ){
         //check if the mapped (so blockids in the dimcension we are computing) is contiguous or not.
         if(id > 0 && d_vmesh->sortedBlockMappedGID[id-1] != d_vmesh->sortedBlockMappedGID[id] - 1 )
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
                          h_vmesh->sortedBlockMappedGID, h_vmesh->sortedBlockMappedGID + h_vmesh->size(),
                          h_vmesh->sortedBlockLID);
      vmesh::prepareColumnCompute<<<cuGridSize, cuBlockSize, 0, stream>>>(d_vmesh);
      LID *newEnd = thrust::remove_if(thrust::cuda::par.on(stream),
                                      h_vmesh->columnStartLID + 1,
                                      h_vmesh->columnStartLID + h_vmesh->size(),
                                      isZero()
                                      );
      h_vmesh->nColumns = newEnd - h_vmesh->columnStartLID;       
      h_vmesh->sortDimension = dimension;
      //copy nColumns & sortDimension to device version
      cudaMemcpy(d_vmesh, h_vmesh, sizeof(VelocityMeshCuda<GID, LID>), cudaMemcpyHostToDevice);      


      
      
      const bool debugSort=true;
      if(debugSort) {
         //print out result of sort
         GID *blockIDs = (GID *) malloc(sizeof(GID) *  h_vmesh->size());
         GID *sortedBlockMappedGID = (GID *) malloc(sizeof(GID) *  h_vmesh->size());
         LID *sortedBlockLID = (LID *) malloc(sizeof(LID) * h_vmesh->size());
         LID *columnStartLID = (LID *) malloc(sizeof(LID) * h_vmesh->size());
         
         cudaMemcpyAsync(blockIDs, h_vmesh->blockIDs , sizeof(GID) * h_vmesh->size() , cudaMemcpyDeviceToHost, stream);
         cudaMemcpyAsync(sortedBlockMappedGID, h_vmesh->sortedBlockMappedGID , sizeof(GID) * h_vmesh->size() , cudaMemcpyDeviceToHost, stream);
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
            LID indices[3];
            h_vmesh->getIndices(blockIDs[sortedBlockLID[i]], indices);
            printf("%d %d %d,%d,%d\n", sortedBlockMappedGID[i], blockIDs[sortedBlockLID[i]], indices[0], indices[1], indices[2]);
         }            
         printf("--------------------------------------------\n");

         free(blockIDs);
         free(sortedBlockMappedGID);
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
