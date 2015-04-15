#include "../velocity_mesh_cuda.h"


bool accelerate_velocity_mesh_cuda(Realf *blockdata, vmesh::GlobalID *blocks, uint nBlocks ){
   vmesh::initVelocityMeshCuda(blockdata, blocks, nBlocks);
//   cudaDeviceSynchronize();
}
