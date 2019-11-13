#include "reed_solomon.h"


int main(int argc, const char* argv[]) {
    rs_context context = {0};
    size_t len = 1024;
    size_t num = 100;
    uint8_t* ins = (uint8_t*)calloc(len, num);
    uint8_t* outs = (uint8_t*)calloc(len, num);

    reed_solomon_encode(&context, ins, outs, len, num);

    free(ins);
    free(outs);
}
