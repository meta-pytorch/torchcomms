#include <stdio.h>
#include <papi.h>
#include <stdlib.h>
#include "cat_arch.h"

#define ITER 1

void papi_stop_and_print_placeholder(long long theory, FILE *fp);
void papi_stop_and_print(long long theory, int EventSet, FILE *fp);

// Non-FMA-like computations.
#if defined(ARM)
half test_hp_scalar_VEC_24( uint64 iterations, int EventSet, FILE *fp );
half test_hp_scalar_VEC_48( uint64 iterations, int EventSet, FILE *fp );
half test_hp_scalar_VEC_96( uint64 iterations, int EventSet, FILE *fp );
#else
float test_hp_scalar_VEC_24( uint64 iterations, int EventSet, FILE *fp );
float test_hp_scalar_VEC_48( uint64 iterations, int EventSet, FILE *fp );
float test_hp_scalar_VEC_96( uint64 iterations, int EventSet, FILE *fp );
#endif

float test_sp_scalar_VEC_24( uint64 iterations, int EventSet, FILE *fp );
float test_sp_scalar_VEC_48( uint64 iterations, int EventSet, FILE *fp );
float test_sp_scalar_VEC_96( uint64 iterations, int EventSet, FILE *fp );

double test_dp_scalar_VEC_24( uint64 iterations, int EventSet, FILE *fp );
double test_dp_scalar_VEC_48( uint64 iterations, int EventSet, FILE *fp );
double test_dp_scalar_VEC_96( uint64 iterations, int EventSet, FILE *fp );

// Functions to emulate FMA.
#if defined(ARM)
half test_hp_scalar_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
half test_hp_scalar_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
half test_hp_scalar_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
#else
float test_hp_scalar_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
float test_hp_scalar_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
float test_hp_scalar_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
#endif

float test_sp_scalar_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
float test_sp_scalar_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
float test_sp_scalar_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );

double test_dp_scalar_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
double test_dp_scalar_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
double test_dp_scalar_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );

