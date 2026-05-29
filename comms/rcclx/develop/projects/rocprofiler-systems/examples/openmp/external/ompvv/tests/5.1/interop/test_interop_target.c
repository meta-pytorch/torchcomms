//===--------------------- test_interop_target.c ----------------------------===//
//
// OpenMP API Version 5.1 Nov 2020
// 
// Test for interop construct utilizing target as the interopability. Runs
// a OpenMP target construct and treating it as an external operation to test
// if the interop construct properly executes and returns the result. Test
// passes if the results are processed correctly and fails if not.
//
////===---------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int interopTestTarget() {
  int errors = 0;
  int A[N];
  
  int device = omp_get_default_device();
  omp_interop_t obj = omp_interop_none;

  for (int i = 0; i < N; i++) {
    A[i] = 0;
  }

  #pragma omp interop init(targetsync: obj) device(device) \
      depend(inout: A[0:N]) // Pass array through to the interop and require it
  {
    #pragma omp target depend(inout: A[0:N]) nowait \
      map(tofrom: A[0:N]) device(device)
    for (int j = 0; j < N; j++) {
      #pragma omp atomic
      A[j] += 5;
    }

  }

  #pragma omp interop destroy(obj) depend(out: A[0:N])

  // Interop should be completed and cleaned up by here
  
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, A[i] != 5);
  }

  return errors;
}

int main () {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;

  int numdevices = omp_get_num_devices();

  OMPVV_WARNING_IF(numdevices <= 0, "No devices detected, interop target test running on host");

  errors = interopTestTarget();

  OMPVV_REPORT_AND_RETURN(errors);

  return 0;
}
