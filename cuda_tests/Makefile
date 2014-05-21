# -*- indent-tabs-mode:t; -*-

NVCC = nvcc
CC = mpiCC

EXE_NAME = test

NVCC_FLAGS = -arch=compute_35 -code=sm_35
CCFLAGS = #-lcuda -lcudart
CUDACFLAGS=-I${CUDA_PATH}/include -L$(CUDA_PATH)/lib64

# Libraries
LIBRARY_PREFIX = $(USERAPPL)/openmpi/libraries

INC_BOOST = -I$(LIBRARY_PREFIX)/boost/include
#INC_BOOST = -I~/appl_taito/openmpi/libraries/boost/include
LIB_BOOST = -L$(LIBRARY_PREFIX)/boost/lib -lboost_program_options

LIB_PROFILE = -L$(LIBRARY_PREFIX)/phiprof/lib -lphiprof
INC_PROFILE = -I$(LIBRARY_PREFIX)/phiprof/include

LIBS = $(LIB_BOOST) $(LIB_PROFILE)
INCS = $(INC_BOOST) $(INC_PROFILE)

default: gpu_test.o spatial_cell.o
	$(CC) $(LIBS) $(CCFLAGS) $(CUDAFLAGS) spatial_cell.o gpu_test.o -o $(EXE_NAME)
	
gpu_test.o: gpu_test.cpp gpu_velocity_grid.hpp ../spatial_cell.hpp 
	$(CC) -c gpu_test.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS) ../spatial_cell.hpp

spatial_cell.o: ../spatial_cell.cpp ../spatial_cell.hpp
	$(CC) -c ../spatial_cell.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS)

clean:
	rm -rf *.~ *.o *.ghc
