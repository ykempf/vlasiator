#include "../velocity_mesh_cuda.h"


bool accelerateVelocityMeshCuda(Realf **blockDatas, vmesh::GlobalID **blockIDs, vmesh::LocalID *nBlocks, const uint nCells ){
//   cudaStream_t streams[nCells];   
   for (int i = 0; i < nCells; i++) {
     // TODO, put each cell into a separate stream:      cudaStreamCreate(&streams[i]);
      vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID> *d_vmesh =
         vmesh::createVelocityMeshCuda(blockDatas[i], blockIDs[i], nBlocks[i]);

      vmesh::destroyVelocityMeshCuda(d_vmesh);
   }
   
   cudaDeviceSynchronize();
}
