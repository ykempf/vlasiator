#include "../velocity_mesh_cuda.h"


bool accelerateVelocityMeshCuda(Realf **blockDatas, vmesh::GlobalID **blockIDs, uint *nBlocks, const uint nCells ){
   cudaStream_t streams[nCells];
   
   for (int i = 0; i < nCells; i++) {
//      cudaStreamCreate(&streams[i]);
      vmesh::VelocityMeshCuda<vmesh::GlobalID> *d_vmesh = vmesh::createVelocityMeshCuda(blockDatas[i], blockIDs[i], nBlocks[i]);
      // TODO launch kernel
      //   copy back
      vmesh::destroyVelocityMeshCuda(d_vmesh);

//   cudaDeviceSynchronize();
      }
}
