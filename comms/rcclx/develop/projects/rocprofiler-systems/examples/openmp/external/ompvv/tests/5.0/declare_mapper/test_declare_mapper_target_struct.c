//===--- test_declare_mapper_target_struct.c --- test declare mapper on struct on device --------- -------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This example has been adapted from the 5.0 OpenMP Examples document.
// the declare mapper directive specifies that any structure of type myvec_t for which 
// implicit data-mapping rules apply will be mapped according to its map clause. The 
// variable v is used for referencing the structure and its elements within the map 
// clause. Within the map clause the v variable specifies that all elements of the 
// structure are to be mapped.
//
////===---------------------------------------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

typedef struct myvec{
    size_t len;
    double *data;
} myvec_t;

#pragma omp declare mapper(myvec_t v) \
                    map(v, v.data[0:v.len])

void init( myvec_t *s )
{ 
  for(size_t i = 0; i < s->len; i++)
    s->data[i] = i; 
}

int test_declare_mapper_struct() {

  OMPVV_INFOMSG("test_declare_mapper_struct");
  int errors = 0;

   myvec_t s;

   s.data = (double *)calloc(N,sizeof(double));
   s.len  = N;

#pragma omp target
  {
    init(&s);
  }
  for (int i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET(errors, s.data[i] != i);
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_mapper_struct());

  OMPVV_REPORT_AND_RETURN(errors);

}
