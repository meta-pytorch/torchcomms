//===-- test_target_data_use_device_ptr.c ---------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This file is a test for the use_device_ptr when used with the map clause
// with target data directive. This test uses arrays of size N which values
// are modified on the  device and tested in the host. Once the array has
// been mapped to the device, the use_device_ptr should be able to be used
// with the ptr to the array and subsequent modify values on the device.
// This test focuses on address conversions of use_device_ptr clauses will
// occur as if performed after all variables are mapped according to those
// map clauses.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 100

int main() {
  int errors = 0;
  int *array_device = NULL;
  int *array_host = NULL;

  array_device = (int *)malloc(N*sizeof(int));
  array_host = (int *)malloc(N*sizeof(int));

  for (int i = 0; i < N; ++i){
    array_device[i] = i;
    array_host[i] = 0;
  }
  OMPVV_TEST_OFFLOADING;

#pragma omp target data map(to: array_device[0:N]) use_device_ptr(array_device)
    {
#pragma omp target is_device_ptr(array_device) map(tofrom: array_host[0:N])
      {
        for (int i = 0; i < N; ++i) {
          array_host[i] += *(array_device + i);
        }
      } // end target
    } // end target data

  // checking results
  for (int i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET(errors, array_host[i] != i);
  }

  free(array_device);
  free(array_host);

  OMPVV_REPORT_AND_RETURN(errors);

}
