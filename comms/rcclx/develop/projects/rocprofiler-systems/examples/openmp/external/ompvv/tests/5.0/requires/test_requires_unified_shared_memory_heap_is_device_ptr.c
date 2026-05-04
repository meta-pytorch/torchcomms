//===---test_requires_unified_shared_memory_heap_is_device_ptr.c -----------===//
//
// OpenMP API Version 5.0 Nov 2018
// 
// This test Checks for unified shared memory of an array that is allocated on 
// the heap and that is accessed from host and device with the same pointer
//
// To guarantee the use of the device pointer, we use the is_device_ptr clause
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_shared_memory

int unified_shared_memory_heap() {
  OMPVV_INFOMSG("Unified shared memory testing - Array on heap");
  int errors = 0;
  
  int *anArray;
  int anArrayCopy[N];

  anArray = (int*)malloc(sizeof(int)*N);
  if( anArray == NULL ) {
     OMPVV_ERROR("Memory was not properly allocated");
     OMPVV_RETURN(1);
  }

  for (int i = 0; i < N; i++) {
    anArray[i] = i;
    anArrayCopy[i] = 0;
  }
  // Modify in the device
#pragma omp target is_device_ptr(anArray)
  {
    for (int i = 0; i < N; i++) {
      anArray[i] += 10;
    }
  }
  // Modify again on the host
  for (int i = 0; i < N; i++) {
    anArray[i] += 10;
  }

  // Get the value the device is seeing
#pragma omp target is_device_ptr(anArray)
  {
    for (int i = 0; i < N; i++) {
      anArrayCopy[i] = anArray[i];
    }
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, anArray[i] != i + 20);
    OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy[i] != i + 20);
    if (errors) break;
  }

  free(anArray);
  return errors;
}
int main() {
  int isOffloading;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution");
  int errors = 0;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, unified_shared_memory_heap());

  OMPVV_REPORT_AND_RETURN(errors);
}
