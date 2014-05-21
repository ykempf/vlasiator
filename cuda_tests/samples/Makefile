# editor magic, do not pay attention
# -*- indent-tabs-mode:t; -*-

CUFLAGS = -arch=compute_35 -code=sm_35

bb: bounding_box.cu
	nvcc $(CUFLAGS) -O3 bounding_box.cu -o bb
