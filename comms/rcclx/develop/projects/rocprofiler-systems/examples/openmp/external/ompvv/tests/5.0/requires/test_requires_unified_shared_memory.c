//===---test_requires_unified_shared_memory.c -------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// 
// This test just uses requries unified_shared_memory in a simple target region
// and it is intended to show that this would not break anything in the compiler
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_shared_memory

int unified_shared_memory() {
  OMPVV_INFOMSG("Unified shared memory testing");
  int errors = 0;
  
  int aVariable = 0;
#pragma omp target map(tofrom:aVariable)
  {
    aVariable += 10;
  }

  aVariable += 10;
  OMPVV_TEST_AND_SET_VERBOSE(errors, aVariable != 20);
  return errors;
}

int main() {
  int isOffloading;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution");
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, unified_shared_memory());

  OMPVV_REPORT_AND_RETURN(errors);
}
