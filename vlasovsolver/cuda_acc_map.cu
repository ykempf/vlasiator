#include "../velocity_mesh_cuda.h"

bool accelerateVelocityMeshCuda(Realf **blockDatas, vmesh::GlobalID **blockIDs, vmesh::LocalID *nBlocks,const  vmesh::LocalID gridLength[3], const Realf blockSize[3], const Realf gridMinLimits[3], const uint nCells ){
   cudaStream_t streams[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* d_vmesh[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* h_vmesh[nCells];
   

   /*allocate memory for all cells, these operations are blocking*/
   //TODO: add checks/throttling to make sure there is enough memory on device...

   for (int i = 0; i < nCells; i++) {
      cudaStreamCreate(&streams[i]);
      vmesh::createVelocityMeshCuda(&(d_vmesh[i]), &(h_vmesh[i]), nBlocks[i], gridLength, blockSize, gridMinLimits);
   }
   
   for (int i = 0; i < nCells; i++) {
      vmesh::uploadMeshData(d_vmesh[i], h_vmesh[i], blockDatas[i], blockIDs[i], streams[i]);
      vmesh::sortVelocityBlocksInColumns(d_vmesh[i], h_vmesh[i], 0, streams[i]);
      vmesh::sortVelocityBlocksInColumns(d_vmesh[i], h_vmesh[i], 1, streams[i]);
      vmesh::sortVelocityBlocksInColumns(d_vmesh[i], h_vmesh[i], 2, streams[i]);
   }
   for (int i = 0; i < nCells; i++) {
      vmesh::destroyVelocityMeshCuda(d_vmesh[i], h_vmesh[i]);
      cudaStreamDestroy(streams[i]);
   }

   //cudaDeviceSynchronize();
}
