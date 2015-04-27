#include "../velocity_mesh_cuda.h"


bool accelerateVelocityMeshCuda(Realf **blockDatas, vmesh::GlobalID **blockIDs, vmesh::LocalID *nBlocks, const uint nCells ){
   cudaStream_t streams[nCells];
   for (int i = 0; i < nCells; i++) {
      cudaStreamCreate(&streams[i]);
   }
   
   for (int i = 0; i < nCells; i++) {
      vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID> *d_vmesh =
         vmesh::createVelocityMeshCuda(blockDatas[i], blockIDs[i], nBlocks[i], streams[i]);
      int blockSize = 512; 
      int gridSize = 1 + nBlocks[i] / blockSize; // value determine by block size and total work
      cudaStreamSynchronize(streams[i]);
      vmesh::initBlockOffsets<<<gridSize, blockSize, 0, streams[i]>>>(d_vmesh);
      cudaStreamSynchronize(streams[i]);
      vmesh::testBlockOffsets<<<gridSize, blockSize, 0, streams[i]>>>(d_vmesh, 610);
      
      vmesh::destroyVelocityMeshCuda(d_vmesh, streams[i]);
   }


   for (int i = 0; i < nCells; i++) {
      cudaStreamDestroy(streams[i]);
   }


}
