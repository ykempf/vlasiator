#include "../velocity_mesh_cuda.h"




bool accelerateVelocityMeshCuda(Realf **blockDatas,
                                vmesh::GlobalID **blockIDs,
                                vmesh::LocalID *nBlocks,
                                const  vmesh::LocalID gridLength[3],
                                const Realf blockSize[3],
                                const Realf gridMinLimits[3],
                                const uint nCells,
                                const Realf intersection, const Realf intersection_di, const Realf intersection_dj, const Realf intersection_dk,
                                uint dimension
                                ){
   cudaStream_t streams[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* d_sourceVmesh[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* h_sourceVmesh[nCells];
   
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* d_targetVmesh[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* h_targetVmesh[nCells];


   /*allocate memory for all cells, these operations are blocking*/
   //TODO: add checks/throttling to make sure there is enough memory on device...

   for (int i = 0; i < nCells; i++) {
      cudaStreamCreate(&streams[i]);
      vmesh::createVelocityMeshCuda(&(d_sourceVmesh[i]), &(h_sourceVmesh[i]), nBlocks[i], gridLength, blockSize, gridMinLimits);
   }
   
   for (int i = 0; i < nCells; i++) {
      vmesh::uploadMeshData(d_sourceVmesh[i], h_sourceVmesh[i], blockDatas[i], blockIDs[i], streams[i]);
      vmesh::sortVelocityBlocksInColumns(d_sourceVmesh[i], h_sourceVmesh[i], 1, streams[i]);
      /*
      vmesh::createTargetMesh(d_targetVmesh[i], h_targetVmesh[i],
                              d_sourceVmesh[i], h_sourceVmesh[i],
                              intersection, intersection_di, intersection_dj, intersection_dk,
                              dimension, streams[i]);
      */
//      vmesh::sortVelocityBlocksInColumns(d_sourceVmesh[i], h_sourceVmesh[i], 1, streams[i]);
//      vmesh::sortVelocityBlocksInColumns(d_sourceVmesh[i], h_sourceVmesh[i], 2, streams[i]);
   }
   for (int i = 0; i < nCells; i++) {
      vmesh::destroyVelocityMeshCuda(d_sourceVmesh[i], h_sourceVmesh[i]);
      cudaStreamDestroy(streams[i]);
   }

   //cudaDeviceSynchronize();
}

