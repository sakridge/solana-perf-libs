#include <inttypes.h>
#include "gpu_common.h"
#include "reed_solomon.h"

#define USE_CLOCK_GETTIME
#include "perftime.h"

uint8_t __device__ mul_table[256];


void __device__ gal_mul(const uint8_t* in, uint8_t* out, const size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = mul_table[in[i]];
    }
}

void __device__ gal_mul_xor(const uint8_t* in, uint8_t* out, const size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] ^= mul_table[in[i]];
    }
}

void __global__ reed_solomon_encode_kernel(const uint8_t* ins, uint8_t* outs, const size_t len, size_t num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num) {
        const uint8_t* in = &ins[i * len];
        uint8_t* out = &outs[i * len];
        gal_mul(in, out, len);
    }
}

void reed_solomon_encode(rs_context* context, uint8_t* ins, uint8_t* outs, size_t len, size_t num) {
    size_t num_threads_per_block = 64;
    size_t total_coding_blocks = len * num;
    int num_blocks = ROUND_UP_DIV(total_coding_blocks, num_threads_per_block);

    cudaStream_t stream = 0;

    uint8_t* ins_device = NULL;
    CUDA_CHK(cudaMalloc(&ins_device, total_coding_blocks));

    uint8_t* outs_device = NULL;
    CUDA_CHK(cudaMalloc(&outs_device, total_coding_blocks));

    CUDA_CHK(cudaMemcpyAsync(ins_device, ins, total_coding_blocks, cudaMemcpyHostToDevice, stream));

    perftime_t start, end;
    get_time(&start);

    reed_solomon_encode_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(
            ins_device, outs_device, len, num);
    CUDA_CHK(cudaPeekAtLastError());

    CUDA_CHK(cudaMemcpyAsync(outs, outs_device, total_coding_blocks, cudaMemcpyDeviceToHost, stream));

    CUDA_CHK(cudaStreamSynchronize(stream));

    get_time(&end);
    LOG("time diff: %f\n", get_diff(&start, &end));
    cudaFree(ins_device);
    cudaFree(outs_device);
}
