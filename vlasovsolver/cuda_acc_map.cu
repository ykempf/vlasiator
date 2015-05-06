#include "../velocity_mesh_cuda.h"

bool accelerateVelocityMeshCuda(Realf **blockDatas, vmesh::GlobalID **blockIDs, vmesh::LocalID *nBlocks,const  vmesh::LocalID gridLength[3], const Real blockSize[3], const uint nCells ){
   cudaStream_t streams[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* d_vmesh[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* h_vmesh[nCells];
   
   /*convert Real -> Realf, GPU version is completely in Realf (float most likely)*/
   const Realf blockSize_f[3] = {(Realf) blockSize[0] / WID, (Realf) blockSize[1] / WID, (Realf) blockSize[2] / WID};

   /*allocate memory for all cells, these operations are blocking*/
   //TODO: add checks/throttling to make sure there is enough memory on device...

   for (int i = 0; i < nCells; i++) {
      cudaStreamCreate(&streams[i]);
      vmesh::createVelocityMeshCuda(&(d_vmesh[i]), &(h_vmesh[i]), nBlocks[i], gridLength, blockSize_f);
   }
   
   for (int i = 0; i < nCells; i++) {
      vmesh::uploadMeshData(d_vmesh[i], h_vmesh[i], blockDatas[i], blockIDs[i], streams[i]);
      vmesh::sortVelocityBlocks(d_vmesh[i], h_vmesh[i], 0, streams[i]);
      vmesh::sortVelocityBlocks(d_vmesh[i], h_vmesh[i], 1, streams[i]);
      vmesh::sortVelocityBlocks(d_vmesh[i], h_vmesh[i], 2, streams[i]);
   }
   for (int i = 0; i < nCells; i++) {
      vmesh::destroyVelocityMeshCuda(d_vmesh[i], h_vmesh[i]);
      cudaStreamDestroy(streams[i]);
   }

   //cudaDeviceSynchronize();
}
