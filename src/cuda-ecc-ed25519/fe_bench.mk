V?=debug
include ../gpu-common.mk

all: fe_bench

CFLAGS+=-I../common -Xcompiler "-Wno-error=unused-function"

fe_bench: fe_bench.cu fe.cu fe.h
	$(NVCC) -o $@ --ptxas-options=-v $(CFLAGS) $(GPU_CFLAGS) $< 
