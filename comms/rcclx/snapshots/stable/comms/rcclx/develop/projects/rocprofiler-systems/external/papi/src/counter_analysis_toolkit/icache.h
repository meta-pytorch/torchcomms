#ifndef _INSTR_CACHE_
#define _INSTR_CACHE_

#include <stdio.h>
#include "hw_desc.h"

#define NO_COPY 0
#define DO_COPY 1

#define FALSE_IF 0
#define TRUE_IF  1

#define COLD_RUN   0
#define NORMAL_RUN 1

#define BUF_ELEM_CNT 32*1024*1024 // Hopefully larger than the L3 cache.

#define RNG() {\
    b  = ((z1 << 6) ^ z1) >> 13;\
    z1 = ((z1 & 4294967294U) << 18) ^ b;\
    b  = ((z2 << 2) ^ z2) >> 27;\
    z2 = ((z2 & 4294967288U) << 2) ^ b;\
    b  = ((z3 << 13) ^ z3) >> 21;\
    z3 = ((z3 & 4294967280U) << 7) ^ b;\
    b  = ((z4 << 3) ^ z4) >> 12;\
    z4 = ((z4 & 4294967168U) << 13) ^ b;\
    b  = ((z1 << 6) ^ z4) >> 13;\
    z1 = ((z1 & 4294967294U) << 18) ^ b;\
    b  = ((z2 << 2) ^ z1) >> 27;\
    b += z4;\
    z2 = ((z2 & 4294967288U) << 2) ^ b;\
    result = z1 ^ z2 ^ z3 ^ z4;\
}

void i_cache_driver(char* papi_event_name, int init, hw_desc_t *hw_desc, char* outdir, int show_progress);
void seq_driver(FILE* ofp_papi, char* papi_event_name, int init, int show_progress);

#endif
