V?=debug
include ../gpu-common.mk

all: ge5_bench

CFLAGS+=-I../common -Xcompiler "-Wno-error=unused-function"

ge5_bench: ge5_bench.cu ge.cu ge.h fe.cu fe.h
	$(NVCC) -o $@ -lineinfo --ptxas-options=-v $(CFLAGS) $(GPU_CFLAGS) $< 
