# editor magic, do not pay attention
# -*- indent-tabs-mode:t; -*-

CUFLAGS = -arch=compute_35 -code=sm_35

bb: reduce.cu culock.hpp
	nvcc $(CUFLAGS) -O3 reduce.cu -o bb
