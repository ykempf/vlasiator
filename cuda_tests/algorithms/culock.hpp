#ifndef lock_hpp
#define lock_hpp

#include <cuda_runtime.h>

#define _LOCK_OPEN 0
#define _LOCK_CLOSED 1

/* 
 * A locking (aka mutex) mechanism to enable atomicity in Cuda kernels and device functions.
 * Usage: Initialise, at start of critical part get lock with lock() and at the end release it with unlock().
 *
 */
class Lock {
private:
    int *state;
    
public:
    Lock(void) {
        int init = _LOCK_OPEN;
        cudaMalloc(&state, sizeof(int));
        cudaMemcpy(state, &init, sizeof(int), cudaMemcpyHostToDevice);
    }
    ~Lock(void) {
        cudaFree(state);
    }
    
    // Get lock
    __device__ void lock(void) {
        while (atomicCAS(state, _LOCK_OPEN, _LOCK_CLOSED) != _LOCK_OPEN);
    }
    
    // Release lock
    __device__ void unlock(void) {
        atomicExch(state, _LOCK_OPEN);
    }

};

#endif
