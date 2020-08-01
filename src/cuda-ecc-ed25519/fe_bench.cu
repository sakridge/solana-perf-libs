#include "fe.cu"
#include "gpu_common.h"
#define USE_CLOCK_GETTIME
#include "perftime.h"

bool g_verbose = true;

__global__ void fe_kernel(fe* hs, const fe* fs, const fe* gs, uint64_t total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        fe_mul(hs[i], fs[i], gs[i]);
    }
}

int main(int argc, const char* argv[]) {
    int arg = 1;
    uint64_t total = strtol(argv[arg++], NULL, 10);

    size_t size = sizeof(fe) * total;
    fe* hs_h = (fe*)calloc(size, 1);
    fe* fs_h = (fe*)calloc(size, 1);
    fe* gs_h = (fe*)calloc(size, 1);

    fe* hs_d = NULL;
    fe* fs_d = NULL;
    fe* gs_d = NULL;

    CUDA_CHK(cudaMalloc(&hs_d, size));
    CUDA_CHK(cudaMalloc(&fs_d, size));
    CUDA_CHK(cudaMalloc(&gs_d, size));

    cudaStream_t stream = {};
    CUDA_CHK(cudaStreamCreate(&stream));

    CUDA_CHK(cudaMemcpyAsync(hs_d, hs_h, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(fs_d, hs_h, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(gs_d, hs_h, size, cudaMemcpyHostToDevice, stream));

    int num_threads_per_block = 64;
    int num_blocks = ROUND_UP_DIV(total, num_threads_per_block);

    perftime_t start, end;
    get_time(&start);
    for (int i = 0; i < 10; i++) {
        fe_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
                (hs_d,
                 fs_d,
                 gs_d,
                 total);
        CUDA_CHK(cudaPeekAtLastError());
    }

    cudaError_t err = cudaMemcpyAsync(hs_h, hs_d, size, cudaMemcpyDeviceToHost, stream);
    CUDA_CHK(err);

    CUDA_CHK(cudaStreamSynchronize(stream));

    get_time(&end);
    LOG("time diff: %.2f us\n", get_diff(&start, &end));

    free(hs_h);
}
