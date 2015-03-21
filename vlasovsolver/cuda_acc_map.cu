#include "../velocity_mesh_cuda.h"


bool accelerate_velocity_mesh_cuda(Realf *blockdata, vmesh::GlobalID *blocks, uint nBlocks ){
   vmesh::readInMesh<<<1,1>>>(blockdata, blocks, nBlocks);
   cudaDeviceSynchronize();
}
