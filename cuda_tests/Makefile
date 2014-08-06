# -*- indent-tabs-mode:t; -*-
include Makefile.${VLASIATOR_ARCH}

EXE_NAME = test.out

DEBUG = -g
CUDADEBUG = -G


NVCC_FLAGS += -Xcompiler -march=native -O0 -lmpi $(CUDADEBUG) 
# Next line fixes a clash with nvcc parser and boost
NVCC_FLAGS +=-DBOOST_NOINLINE='__attribute__ ((noinline))'
# This enables linking across multiple cuda files. A separate nvcc linking is needed for this (gpu_link.o).
NVCC_FLAGS +=--relocatable-device-code=true
CCFLAGS += -O3 -std=c++11 -Wno-write-strings -fpermissive
CCFLAGS += -DUSE_AGNER_VECTORCLASS -DACC_SEMILAG_PPM 

LIBS += $(LIB_BOOST) $(LIB_PROFILE) $(LIB_MPI) $(LIB_ZOLTAN)
INCS += $(INC_BOOST) $(INC_PROFILE) $(SOURCE_FOLDER) $(INC_MPI) $(INC_EIGEN) $(INC_DCCRG) $(INC_ZOLTAN) -I../vlasovsolver/ -I../ $(INC_VECTORCLASS)

OBJS = spatial_cell.o gpu_velocity_grid.o gpu_test.o spatial_cell_funcs.o gpu_link.o bounding_box.o
CPU_OBJS = cpu_test.o spatial_cell.o spatial_cell_funcs.o

default: $(OBJS)
	$(CC) $(DEBUG) $(OBJS) $(LIBS) $(CCFLAGS) $(CUDAFLAGS) -o $(EXE_NAME)

# CUDA agnostic CPU only version
cpu: $(CPU_OBJS)
	$(CC) $(CCFLAGS) $(DEBUG) $(CPU_OBJS) -o $(EXE_NAME) $(LIBS)

cpu_test.o: cpu_test.cpp gpu_velocity_grid.hpp spatial_cell.o
	$(CC) $(DEBUG) $(INCS) $(CCFLAGS) -c cpu_test.cpp

gpu_test.o: gpu_test.cpp gpu_velocity_grid.hpp spatial_cell.o spatial_cell_funcs.o
	$(CC) $(DEBUG) -c gpu_test.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS)

gpu_velocity_grid.o: gpu_velocity_grid.cu gpu_velocity_grid.hpp bounding_box.cu spatial_cell.o
	$(NVCC) $(DEBUG) $(NVCC_FLAGS) $(INCS) $(INC_MPI) -c gpu_velocity_grid.cu

spatial_cell.o: ../spatial_cell.cpp ../spatial_cell.hpp
	$(CC) $(DEBUG) -c ../spatial_cell.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS)

spatial_cell_funcs.o: spatial_cell_funcs.cpp gpu_velocity_grid.hpp spatial_cell_funcs.hpp acceleration/cpu_acc_semilag_full.hpp acceleration/map_3d.hpp
	$(CC) $(DEBUG) $(INCS) $(CCFLAGS) $(CUDAFLAGS) -c spatial_cell_funcs.cpp

bounding_box.o: bounding_box.cu gpu_velocity_grid.hpp
	$(NVCC) $(DEBUG) $(NVCC_FLAGS) $(INCS) $(INC_MPI) -c bounding_box.cu

gpu_link.o: gpu_velocity_grid.o bounding_box.o gpu_velocity_grid.hpp
	$(NVCC) -dlink $(DEBUG) $(LIBS) $(NVCC_FLAGS) $(CUDAFLAGS) gpu_velocity_grid.o bounding_box.o -o gpu_link.o

clean:
	rm -rf *~ *.o *.ghc

