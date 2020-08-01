V?=debug
include ../gpu-common.mk

all: ge3_bench

CFLAGS+=-I../common -Xcompiler "-Wno-error=unused-function"

ge3_bench: ge3_bench.cu ge.cu ge.h fe.cu fe.h
	$(NVCC) -o $@ -lineinfo --ptxas-options=-v $(CFLAGS) $(GPU_CFLAGS) $< 
