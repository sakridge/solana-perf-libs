#include "fe.cu"
#include "ge.cu"
#include "gpu_common.h"
#define USE_CLOCK_GETTIME
#include "perftime.h"

bool g_verbose = true;

#define SCALAR_SIZE 32
__global__ void ge3_kernel(
        const unsigned char* a, //32-byte
        uint8_t* out,
        uint64_t total
        ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        ge_p3 r;
        ge_scalarmult_base(&r, &a[i * SCALAR_SIZE]);
        out[i] = r.X[0] ^ r.Y[1] ^ r.Z[7];
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("invalid args\n");
        return 1;
    }
    int arg = 1;
    uint64_t total = strtol(argv[arg++], NULL, 10);
    int num_threads_per_block  = strtol(argv[arg++], NULL, 10);

    size_t ab_size = SCALAR_SIZE * total;
    unsigned char* a_h = (unsigned char*)calloc(ab_size, 1);
    unsigned char* out_h = (unsigned char*)calloc(total, 1);
    size_t p3_size = sizeof(ge_p3) * total;

    for (uint i = 0; i < ab_size; i++) {
        a_h[i] = rand();
    }

    unsigned char* a_d = NULL;
    uint8_t* out_d = NULL;

    CUDA_CHK(cudaMalloc(&a_d, ab_size));
    CUDA_CHK(cudaMalloc(&out_d, total));

    cudaStream_t stream = {};
    CUDA_CHK(cudaStreamCreate(&stream));

    CUDA_CHK(cudaMemcpyAsync(a_d, a_h, ab_size, cudaMemcpyHostToDevice, stream));

    int num_blocks = ROUND_UP_DIV(total, num_threads_per_block);

    perftime_t start, end;
    get_time(&start);
    //for (int i = 0; i < 10; i++) {
        ge3_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
                (a_d,
                 out_d,
                 total);
        CUDA_CHK(cudaPeekAtLastError());
    //}

    cudaError_t err = cudaMemcpyAsync(out_h, out_d, total, cudaMemcpyDeviceToHost, stream);
    CUDA_CHK(err);

    CUDA_CHK(cudaStreamSynchronize(stream));

    get_time(&end);
    LOG("time diff: %.2f us\n", get_diff(&start, &end));

    CUDA_CHK(cudaFree(a_d));
    CUDA_CHK(cudaFree(out_d));
    free(a_h);
    free(out_h);
}
