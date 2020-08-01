V?=debug
include ../gpu-common.mk

all: ge4_bench

CFLAGS+=-I../common -Xcompiler "-Wno-error=unused-function"

ge4_bench: ge4_bench.cu ge.cu ge.h fe.cu fe.h
	$(NVCC) -o $@ -lineinfo --ptxas-options=-v $(CFLAGS) $(GPU_CFLAGS) $< 
