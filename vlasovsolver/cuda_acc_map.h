bool map3DCuda(Realf **blockDatas,
               vmesh::GlobalID **blockIDs,
               vmesh::LocalID *nBlocks,
               Real *intersections,
               const uint nCells,
               const Realf blockSize[3],
               const  vmesh::LocalID gridLength[3],
               const Realf gridMinLimits[3]);