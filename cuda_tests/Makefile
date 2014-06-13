# -*- indent-tabs-mode:t; -*-

include Makefile.${VLASIATOR_ARCH}

EXE_NAME = test.out

DEBUG = -g

LIBS = $(LIB_BOOST) $(LIB_PROFILE)
INCS = $(INC_BOOST) $(INC_PROFILE) $(SOURCE_FOLDER)

OBJS = spatial_cell.o gpu_velocity_grid.o gpu_test.o spatial_cell_funcs.o gpu_link.o bounding_box.o
CPU_OBJS = cpu_test.o spatial_cell.o spatial_cell_funcs.o

default: $(OBJS)
	$(CC) $(DEBUG) $(LIBS) $(CCFLAGS) $(CUDAFLAGS) $(OBJS) -o $(EXE_NAME)

# CUDA agnostic CPU only version
cpu: $(CPU_OBJS)
	$(CC) $(DEBUG) $(CPU_OBJS) -o $(EXE_NAME) $(LIBS)

cpu_test.o: cpu_test.cpp gpu_velocity_grid.hpp ../spatial_cell.hpp
	$(CC) $(DEBUG) $(INCS) $(CCFLAGS) -c cpu_test.cpp

gpu_test.o: gpu_test.cpp gpu_velocity_grid.hpp ../spatial_cell.hpp 
	$(CC) $(DEBUG) -c gpu_test.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS) ../spatial_cell.hpp

gpu_velocity_grid.o: gpu_velocity_grid.cu gpu_velocity_grid.hpp bounding_box.cu
	$(NVCC) $(DEBUG) $(NVCC_FLAGS) $(INCS) $(INC_MPI) -c gpu_velocity_grid.cu

spatial_cell.o: ../spatial_cell.cpp ../spatial_cell.hpp
	$(CC) $(DEBUG) -c ../spatial_cell.cpp $(INCS) $(CCFLAGS) $(CUDAFLAGS)

spatial_cell_funcs.o: spatial_cell_funcs.cpp gpu_velocity_grid.hpp
	$(CC) $(DEBUG) $(INCS) $(CCFLAGS) $(CUDAFLAGS) -c spatial_cell_funcs.cpp

bounding_box.o: bounding_box.cu gpu_velocity_grid.hpp
	$(NVCC) $(DEBUG) $(NVCC_FLAGS) $(INCS) $(INC_MPI) -c bounding_box.cu

gpu_link.o: gpu_velocity_grid.o bounding_box.o gpu_velocity_grid.hpp
	$(NVCC) -dlink $(DEBUG) $(LIBS) $(NVCC_FLAGS) $(CUDAFLAGS) gpu_velocity_grid.o bounding_box.o -o gpu_link.o

clean:
	rm -rf *~ *.o *.ghc

