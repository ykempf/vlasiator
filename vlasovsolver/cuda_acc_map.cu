#include "../velocity_mesh_cuda.h"
 
bool map3DCuda(Realf **blockDatas,
               vmesh::GlobalID **blockIDs,
               vmesh::LocalID *nBlocks,
               Real *intersections,
               const uint nCells,
               const Realf blockSize[3],
               const vmesh::LocalID gridLength[3],
               const Realf gridMinLimits[3]){
   cudaStream_t streams[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* d_sourceVmesh[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* h_sourceVmesh[nCells];
   
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* d_targetVmesh[nCells];
   vmesh::VelocityMeshCuda<vmesh::GlobalID, vmesh::LocalID>* h_targetVmesh[nCells];


   /*allocate memory for all cells, these operations are blocking*/
   //TODO: add checks/throttling to make sure there is enough memory on device...

   for (int i = 0; i < nCells; i++) {
      cudaStreamCreate(&streams[i]);
      vmesh::createVelocityMeshCuda(&(d_sourceVmesh[i]), &(h_sourceVmesh[i]), 
            nBlocks[i], gridLength, blockSize, gridMinLimits);
   }
   
   for (int i = 0; i < nCells; i++) {
      vmesh::uploadMeshData(d_sourceVmesh[i], h_sourceVmesh[i], blockDatas[i], blockIDs[i], streams[i]);
     //Order Z X Y
      //DO Z
      vmesh::sortVelocityBlocksInColumns(d_sourceVmesh[i], h_sourceVmesh[i], 2, streams[i]);
      vmesh::createTargetMesh(&(d_targetVmesh[i]), &(h_targetVmesh[i]),
                              d_sourceVmesh[i], h_sourceVmesh[i],
                              intersections + i * 12,
                              2, streams[i]);
      
//      vmesh::sortVelocityBlocksInColumns(d_sourceVmesh[i], h_sourceVmesh[i], 1, streams[i]);
//      vmesh::sortVelocityBlocksInColumns(d_sourceVmesh[i], h_sourceVmesh[i], 2, streams[i]);
   }
   for (int i = 0; i < nCells; i++) {
      vmesh::destroyVelocityMeshCuda(d_sourceVmesh[i], h_sourceVmesh[i]);
      cudaStreamDestroy(streams[i]);
   }

   //cudaDeviceSynchronize();
}

