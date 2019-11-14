#include "reed_solomon.h"
#include "gpu_common.h"


int main(int argc, const char* argv[]) {
    g_verbose = true;
    rs_context context = {0};
    context.num_data = 32;
    context.num_coding = 32;
    size_t len = 1024;
    size_t num_sessions = 1000;

    uint8_t* ins = (uint8_t*)calloc(len, num_sessions * context.num_data);
    uint8_t* outs = (uint8_t*)calloc(len, num_sessions * context.num_coding);

    for (size_t i = 0; i < num_sessions * context.num_data; i++) {
        ins[i] = rand() % 255;
    }

    reed_solomon_encode(&context, ins, outs, len, num_sessions);

    free(ins);
    free(outs);
}
