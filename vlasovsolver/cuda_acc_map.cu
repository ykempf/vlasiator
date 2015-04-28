#include "../velocity_mesh_cuda.h"

bool accelerateVelocityMeshCuda(Realf **blockDatas, vmesh::GlobalID **blockIDs, vmesh::LocalID *nBlocks,const  vmesh::LocalID gridLength[3], const Real blockSize[3], const uint nCells ){
   cudaStream_t streams[nCells];
   for (int i = 0; i < nCells; i++) {
      cudaStreamCreate(&streams[i]);
   }

   /*convert Real -> Realf, GPU version is completely in Realf (float most likely)*/
   const Realf blockSize_f[3] = {(Realf) blockSize[0] / WID, (Realf) blockSize[1] / WID, (Realf) blockSize[2] / WID};
 
   for (int i = 0; i < nCells; i++) {
      vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID> *d_vmesh =
         vmesh::createVelocityMeshCuda(blockDatas[i], blockIDs[i], nBlocks[i], gridLength, blockSize_f, streams[i]);
      vmesh::sortVelocityBlocks(d_vmesh, 0, streams[i]);
      vmesh::sortVelocityBlocks(d_vmesh, 1, streams[i]);
      vmesh::sortVelocityBlocks(d_vmesh, 2, streams[i]);

      vmesh::destroyVelocityMeshCuda(d_vmesh, streams[i]);
   }
   for (int i = 0; i < nCells; i++) {
      cudaStreamDestroy(streams[i]);
   }

   //cudaDeviceSynchronize();
}
