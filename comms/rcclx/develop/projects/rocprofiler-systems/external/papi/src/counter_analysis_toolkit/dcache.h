#ifndef _DATA_CACHE_
#define _DATA_CACHE_

#include <stdio.h>
#include <omp.h>
#include "hw_desc.h"
#include "params.h"

#define FACTOR 12LL

int varyBufferSizes(long long *values, double **rslts, double **counter, cat_params_t params, hw_desc_t *hw_desc, long long line_size_in_bytes, float pages_per_block, int pattern, int latency_only, int mode, int ONT);
int get_thread_count();
void d_cache_driver(char* papi_event_name, cat_params_t params, hw_desc_t *hw_desc, int latency_only, int mode);
int d_cache_test(int pattern, cat_params_t params, hw_desc_t *hw_desc, long long stride_in_bytes, float pages_per_block, char* papi_event_name, int latency_only, int mode, FILE* ofp);

#endif
