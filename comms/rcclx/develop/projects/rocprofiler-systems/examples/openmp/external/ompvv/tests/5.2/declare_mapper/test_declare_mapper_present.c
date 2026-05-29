//===---- test_declare_mapper_present.c -------------------------------------===//
// 
// OpenMP API Version 5.2 
//
// This example has been adapted from the 5.2 OpenMP Examples document:
// "declare mapper Directive", and "OpenMP Directive Syntax -> complex iterator"
// The declare mapper directive will be used to automatically map variables
// according to its prescription:  full structure, plus the dynamic storage of the
// data element. These variable should be present on the device.
//
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

typedef struct myvec{
    size_t len;
    int *data;
} myvec_t;

// In addition to the present modifier, the always is requiered
// Given that the present and default modifiers are to be tested
// the struct has to be in the GPU prior to the default mapper
// is tested, therefore the "omp target enter data" with "to"
// is used. Furthermore, given that it is the "from" that is to be
// checked, the "omp target exit data" is set to "delete"
// in this scenario if the always is not on the declare mapper
// the "omp target" is inside the "omp target enter data"
// with the "map to", which would prevent the effect of the
// declared mapped to have effects outside. Therefore, by adding
// the always, the mapper force the update of the declared mapper
// to be seen outside the "target data region"

#pragma omp declare mapper(default:myvec_t v) map(always, present, from: v, v.len, v.data[0:v.len]) 


void init( myvec_t *s )
{ 
  for(int i = 0; i < s->len; i++)
    s->data[i] = i; 
}


int test_declare_mapper_present() { 

  OMPVV_INFOMSG("test_declare_mapper_present");
  int errors = 0;

  myvec_t s;
  s.data = (int *)calloc(N,sizeof(int));
  s.len  = N;

  #pragma omp target enter data map(to: s.len, s.data[0:s.len])

  #pragma omp target
  {
    init(&s);
  }

  #pragma omp target exit data map(delete: s.len, s.data[0:s.len])

  for (int i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET(errors, s.data[i] != i);
  }	
  
  return errors;
}

int main () {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_mapper_present());

  OMPVV_REPORT_AND_RETURN(errors);
}  
