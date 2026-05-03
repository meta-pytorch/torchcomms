//===---test_requires_unified_shared_memory_static.c ------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// We request the use of unified_shared_memory in this program.
// Checking for static arrays. The array is global and then accessed through 
// a pointer from the host and the device.
//
// We use default mapping of the pointer, which should result in mapping of a zero
// length array.
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_shared_memory

// STATIC ARRAY 
int anArray[N];

int unified_shared_memory_static() {
  OMPVV_INFOMSG("Unified shared memory testing - Static Array");
  int errors = 0;
  
  int anArrayCopy[N];
  int * aPtr = anArray;

  for (int i = 0; i < N; i++) {
    anArray[i] = i;
    anArrayCopy[i] = 0;
  }

  // Test for writes to this variable from device
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
  OMPVV_TEST_AND_SET_VERBOSE(errors, unified_shared_memory_static());

  OMPVV_REPORT_AND_RETURN(errors);
}
