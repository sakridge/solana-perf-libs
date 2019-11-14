#ifndef REED_SOLOMON_H
#define REED_SOLOMON_H

#include <inttypes.h>

typedef struct {
    size_t num_data;
    size_t num_coding;
    uint8_t* matrix;
} rs_context;

void reed_solomon_encode(rs_context* context, uint8_t* ins, uint8_t* outs, size_t len, size_t num);

#endif
