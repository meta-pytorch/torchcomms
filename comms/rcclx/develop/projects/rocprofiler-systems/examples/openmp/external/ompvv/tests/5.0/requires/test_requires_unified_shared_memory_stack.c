//===---test_requires_unified_shared_memory_stack.c ------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Testing memory access from host and device to a variable allocated in the stack
// The variable is accessed through a pointer to guarantee that there is no default
// mapping tofrom: of the stack array
//
// Default mapping of the pointer should translate into the host pointer value, which
// should be accessible from the device
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_shared_memory

int unified_shared_memory_stack() {
  OMPVV_INFOMSG("Unified shared memory testing - Array on stack");
  int errors = 0;
  
  int anArray[N];
  int anArrayCopy[N];
  int * aPtr = anArray;

  for (int i = 0; i < N; i++) {
    anArray[i] = i;
    anArrayCopy[i] = 0;
  }

  // Test for writes to this varriable from device
#pragma omp target 
  {
    for (int i = 0; i < N; i++) {
      aPtr[i] += 10;
    }
  }
  // Modify again on the host
  for (int i = 0; i < N; i++) {
    aPtr[i] += 10;
  }

  // Test for reads to this variable from device
#pragma omp target 
  {
    for (int i = 0; i < N; i++) {
      anArrayCopy[i] = aPtr[i];
    }
  }
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, anArray[i] != i + 20);
    OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy[i] != i + 20);
    if (errors) break;
  }
  return errors;
}

int main() {
  int isOffloading;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution");
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, unified_shared_memory_stack());

  OMPVV_REPORT_AND_RETURN(errors);
}
