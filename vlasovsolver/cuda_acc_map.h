bool accelerateVelocityMeshCuda(Realf **blockDatas,
                                vmesh::GlobalID **blockIDs,
                                vmesh::LocalID *nBlocks,
                                const  vmesh::LocalID gridLength[3],
                                const Realf blockSize[3],
                                const Realf gridMinLimits[3],
                                const uint nCells,
                                const Realf intersection, const Realf intersection_di, const Realf intersection_dj, const Realf intersection_dk,
                                uint dimension
                                );


