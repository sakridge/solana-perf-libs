#include "fe.cu"
#include "ge.cu"
#include "gpu_common.h"
#define USE_CLOCK_GETTIME
#include "perftime.h"

bool g_verbose = true;

__global__ void ge_kernel(ge_p1p1* rs, const ge_p2* p, uint64_t total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        ge_p1p1 rl;
        ge_p2 pl = p[i];
        ge_p2_dbl(&rl, &pl);
        rs[i] = rl;
    }
}

int main(int argc, const char* argv[]) {
    int arg = 1;
    uint64_t total = strtol(argv[arg++], NULL, 10);
    int num_threads_per_block  = strtol(argv[arg++], NULL, 10);

    size_t p1p1_size = sizeof(ge_p1p1) * total;
    ge_p1p1* rs_h = (ge_p1p1*)calloc(p1p1_size, 1);
    size_t p2_size = sizeof(ge_p2) * total;
    ge_p2* ps_h = (ge_p2*)calloc(p2_size, 1);

    ge_p1p1* rs_d = NULL;
    ge_p2* ps_d = NULL;

    CUDA_CHK(cudaMalloc(&rs_d, p1p1_size));
    CUDA_CHK(cudaMalloc(&ps_d, p2_size));

    cudaStream_t stream = {};
    CUDA_CHK(cudaStreamCreate(&stream));

    CUDA_CHK(cudaMemcpyAsync(ps_d, ps_h, p2_size, cudaMemcpyHostToDevice, stream));

    int num_blocks = ROUND_UP_DIV(total, num_threads_per_block);

    perftime_t start, end;
    get_time(&start);
    for (int i = 0; i < 10; i++) {
        ge_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
                (rs_d,
                 ps_d,
                 total);
        CUDA_CHK(cudaPeekAtLastError());
    }

    cudaError_t err = cudaMemcpyAsync(rs_h, rs_d, p2_size, cudaMemcpyDeviceToHost, stream);
    CUDA_CHK(err);

    CUDA_CHK(cudaStreamSynchronize(stream));

    get_time(&end);
    LOG("time diff: %.2f us\n", get_diff(&start, &end));

    CUDA_CHK(cudaFree(rs_d));
    CUDA_CHK(cudaFree(ps_d));
    free(ps_h);
    free(rs_h);
}
