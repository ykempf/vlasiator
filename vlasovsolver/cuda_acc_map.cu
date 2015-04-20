#include "../velocity_mesh_cuda.h"


bool accelerateVelocityMeshCuda(Realf *blockdata, vmesh::GlobalID *blocks, uint nBlocks ){
   vmesh::VelocityMeshCuda<vmesh::GlobalID> *d_vmesh = vmesh::createVelocityMeshCuda(blockdata, blocks, nBlocks);
   // TODO launch kernel
   // copy back
   vmesh::destroyVelocityMeshCuda(d_vmesh);

//   cudaDeviceSynchronize();
}
