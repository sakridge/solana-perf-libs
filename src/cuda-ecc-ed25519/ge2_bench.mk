V?=debug
include ../gpu-common.mk

all: ge2_bench

CFLAGS+=-I../common -Xcompiler "-Wno-error=unused-function"

ge2_bench: ge2_bench.cu ge.cu ge.h fe.cu fe.h
	$(NVCC) -o $@ -lineinfo --ptxas-options=-v $(CFLAGS) $(GPU_CFLAGS) $< 
