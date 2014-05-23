# -*- indent-tabs-mode:t; -*-

NVCC = nvcc
CC = mpiCC

EXE_NAME = test.out

DEBUG = #-g

NVCC_FLAGS = -Xcompiler -march=native -arch=compute_35 -code=compute_35 -O3 -lmpi
# Next line fixes a clash with nvcc parser and boost
NVCC_FLAGS +=-DBOOST_NOINLINE='__attribute__ ((noinline))'
CCFLAGS = -O3 -std=c++11
CUDAFLAGS=-I${CUDA_PATH}/include -L$(CUDA_PATH)/lib64 -lcuda -lcudart

# Libraries
LIBRARY_PREFIX = $(USERAPPL)/openmpi/libraries
MPI_HOME = /appl/opt/openmpi/1.8-cuda/gcc/4.8.2/

INC_BOOST = -I$(LIBRARY_PREFIX)/boost/include
LIB_BOOST = -L$(LIBRARY_PREFIX)/boost/lib -lboost_program_options

LIB_PROFILE = -L$(LIBRARY_PREFIX)/phiprof/lib -lphiprof
INC_PROFILE = -I$(LIBRARY_PREFIX)/phiprof/include

INC_MPI =-I${MPI_HOME}/include


LIBS = $(LIB_BOOST) $(LIB_PROFILE)
INCS = $(INC_BOOST) $(INC_PROFILE)

OBJS = spatial_cell.o gpu_velocity_grid.o gpu_velocity_grid_device.o gpu_test.o

default: spatial_cell.o gpu_velocity_grid.o gpu_velocity_grid_device.o gpu_test.o
	$(CC) $(DEBUG) spatial_cell.o gpu_velocity_grid.o gpu_velocity_grid_device.o gpu_test.o $(LIBS) $(CCFLAGS) $(CUDAFLAGS) -o $(EXE_NAME)

gpu_test.o: gpu_test.cpp gpu_velocity_grid.hpp ../spatial_cell.hpp 
	$(CC) $(DEBUG) -c gpu_test.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS) ../spatial_cell.hpp

gpu_velocity_grid.o: gpu_velocity_grid.cpp gpu_velocity_grid.hpp
	$(CC) $(DEBUG) $(INCS) $(CCFLAGS) $(CUDAFLAGS) -c gpu_velocity_grid.cpp

gpu_velocity_grid_device.o: gpu_velocity_grid_device.cu gpu_velocity_grid.hpp
	$(NVCC) $(DEBUG) $(NVCC_FLAGS) $(INCS) $(INC_MPI) -c gpu_velocity_grid_device.cu

spatial_cell.o: ../spatial_cell.cpp ../spatial_cell.hpp
	$(CC) $(DEBUG) -c ../spatial_cell.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS)

clean:
	rm -rf *~ *.o *.ghc
