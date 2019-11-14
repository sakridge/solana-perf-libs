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

void __global__ reed_solomon_encode_kernel(const uint8_t* ins, uint8_t* outs,
                                           const size_t len,
                                           size_t num_data,
                                           size_t num_coding,
                                           size_t num_sessions,
                                           size_t total_coding_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_coding_bytes) {
        size_t i_session = i / (len * num_coding);
        size_t j_session = i % len;
        size_t data = i_session * num_data * len;
        const uint8_t* in = &ins[data + j_session];
        uint8_t out;

        out = mul_table[in[0]];
        for (size_t j = 1; j < num_data; j++) {
            out ^= mul_table[in[j * len]];
        }
        outs[i] = out;
    }
}

#define V_SIZE 4
#define V_TYPE uint32_t
#define PACKED_STORE(start, in) \
    out[start + 0] = mul_table[in & 0xff]; \
    out[start + 1] = mul_table[(in >> 8) & 0xff]; \
    out[start + 2] = mul_table[(in >> 16) & 0xff]; \
    out[start + 3] = mul_table[(in >> 24) & 0xff]

#define PACKED_XOR(start, in) \
    out[start + 0] ^= mul_table[in & 0xff]; \
    out[start + 1] ^= mul_table[(in >> 8) & 0xff]; \
    out[start + 2] ^= mul_table[(in >> 16) & 0xff]; \
    out[start + 3] ^= mul_table[(in >> 24) & 0xff]


void __global__ reed_solomon_encode_kernel_v(const uint8_t* ins, uint8_t* outs,
                                             const size_t len,
                                             size_t num_data,
                                             size_t num_coding,
                                             size_t num_sessions,
                                             size_t total_coding_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i * V_SIZE) < total_coding_bytes) {
        size_t i_session = i / (len * num_coding);
        size_t j_session = i % len;
        size_t data = i_session * num_data * len;
        const uint8_t* in = &ins[(data + j_session) * V_SIZE];
        uint8_t out[V_SIZE];

        V_TYPE in_v = ((V_TYPE*)in)[0];
        PACKED_STORE(0, in_v);
        //PACKED_STORE(4, in_v.y);
        //PACKED_STORE(8, in_v.z);
        //PACKED_STORE(12, in_v.w);

        for (size_t j = 1; j < num_data; j++) {
            in_v = ((V_TYPE*)&in[j * len])[0];
            PACKED_XOR(0, in_v);
            //PACKED_XOR(4, in_v.y);
            //PACKED_XOR(8, in_v.z);
            //PACKED_XOR(12, in_v.w);
        }

#pragma unroll
        for (size_t j = 0; j < V_SIZE; j++) {
            outs[j] = out[j];
        }
    }
}

void reed_solomon_encode(rs_context* context, uint8_t* ins, uint8_t* outs, size_t len, size_t num_sessions) {
    size_t num_threads_per_block = 128;
    size_t total_coding_bytes = context->num_coding * len * num_sessions;
    size_t num_threads = total_coding_bytes / V_SIZE;
    int num_blocks = ROUND_UP_DIV(total_coding_bytes, num_threads_per_block);

    LOG("Launching %zu threads\n", num_threads);

    size_t total_ins_bytes = num_sessions * context->num_data * len;

    cudaStream_t stream = 0;

    uint8_t* ins_device = NULL;
    CUDA_CHK(cudaMalloc(&ins_device, total_ins_bytes));

    uint8_t* outs_device = NULL;
    CUDA_CHK(cudaMalloc(&outs_device, total_coding_bytes));

    CUDA_CHK(cudaMemcpyAsync(ins_device, ins, total_ins_bytes, cudaMemcpyHostToDevice, stream));

    perftime_t start, end;
    get_time(&start);

    reed_solomon_encode_kernel_v<<<num_blocks, num_threads_per_block, 0, stream>>>(
            ins_device, outs_device, len, context->num_data, context->num_coding, num_sessions, total_coding_bytes);
    CUDA_CHK(cudaPeekAtLastError());

    CUDA_CHK(cudaMemcpyAsync(outs, outs_device, total_coding_bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHK(cudaStreamSynchronize(stream));

    get_time(&end);
    LOG("time diff: %.2f\n", get_diff(&start, &end));
    cudaFree(ins_device);
    cudaFree(outs_device);
}
