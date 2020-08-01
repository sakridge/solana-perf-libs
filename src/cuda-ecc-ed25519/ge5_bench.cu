#include "fe.cu"
#include "ge.cu"
#include "gpu_common.h"
#define USE_CLOCK_GETTIME
#include "perftime.h"

bool g_verbose = true;

void __host__ __device__ ge_genlookup(ge_cached* Ai, const ge_p3* A) {
    ge_p1p1 t;
    ge_p3 u;
    ge_p3 A2;

    ge_p3_to_cached(&Ai[0], A);
    ge_p3_dbl(&t, A);
    ge_p1p1_to_p3(&A2, &t);
    ge_add(&t, &A2, &Ai[0]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[1], &u);
    ge_add(&t, &A2, &Ai[1]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[2], &u);
    ge_add(&t, &A2, &Ai[2]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[3], &u);
    ge_add(&t, &A2, &Ai[3]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[4], &u);
    ge_add(&t, &A2, &Ai[4]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[5], &u);
    ge_add(&t, &A2, &Ai[5]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[6], &u);
    ge_add(&t, &A2, &Ai[6]);
    ge_p1p1_to_p3(&u, &t);
    ge_p3_to_cached(&Ai[7], &u);
}

__global__ void ge_genlookup_kernel(
        const ge_p3* As,
        ge_cached* Ais,
        uint64_t total
        ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        ge_genlookup(&Ais[i], &As[i]);
    }
}

void __host__ __device__ ge_scalarmult_vartime(ge_p2* r, const unsigned char* a, const ge_cached* Ai) {
    signed char aslide[256];
    ge_p1p1 t;
    ge_p3 u;
    int i;
    slide(aslide, a);
    ge_p2_0(r);

    for (i = 255; i >= 0; --i) {
        if (aslide[i]) {
            break;
        }
    }

    for (; i >= 0; --i) {
        ge_p2_dbl(&t, r);

        bool a_gt_zero = aslide[i] > 0;
        bool a_lt_zero = aslide[i] < 0;
        if (a_gt_zero || a_lt_zero) {
            ge_p1p1_to_p3(&u, &t);
            const ge_cached* p = a_gt_zero ? &Ai[aslide[i] / 2] : &Ai[(-aslide[i]) / 2];
            ge_addsub(&t, &u, p, a_gt_zero);
        }

        ge_p1p1_to_p2(r, &t);
    }
}

#define SCALAR_SIZE 32
__global__ void ge5_kernel(
        const unsigned char* a, //32-byte
        const ge_cached* A,
        uint8_t* out,
        uint64_t total
        ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        ge_p2 r;
        ge_scalarmult_vartime(&r, &a[i * SCALAR_SIZE], &A[i]);
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
    size_t cached_size = sizeof(ge_cached) * total;
    ge_p3* As_h = (ge_p3*)calloc(p3_size, 1);
    ge_cached* Ac_h = (ge_cached*)calloc(cached_size, 1);

    for (uint i = 0; i < ab_size; i++) {
        a_h[i] = rand();
    }

    unsigned char* a_d = NULL;
    ge_p3* As_d = NULL;
    ge_cached* Ac_d = NULL;
    uint8_t* out_d = NULL;

    CUDA_CHK(cudaMalloc(&a_d, ab_size));
    CUDA_CHK(cudaMalloc(&As_d, p3_size));
    CUDA_CHK(cudaMalloc(&Ac_d, cached_size));
    CUDA_CHK(cudaMalloc(&out_d, total));

    cudaStream_t stream = {};
    CUDA_CHK(cudaStreamCreate(&stream));

    CUDA_CHK(cudaMemcpyAsync(a_d, a_h, ab_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(As_d, As_h, p3_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(Ac_d, Ac_h, cached_size, cudaMemcpyHostToDevice, stream));

    int num_blocks = ROUND_UP_DIV(total, num_threads_per_block);

    perftime_t start, end;
    get_time(&start);
    ge_genlookup_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
        (As_d,
         Ac_d,
         total);
    CUDA_CHK(cudaPeekAtLastError());
    //for (int i = 0; i < 10; i++) {
        ge5_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
                (a_d,
                 Ac_d,
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
