//===--- test_target_teams_distribute_is_device_ptr.c------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses is_device_ptr clause to indicate that the parameter that is
// given is already an address to the data on the device.  The parameter is set
// using the omp_target_alloc which returns the pointer to the location on the
// target that was allocated.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE_THRESHOLD 512
#define ARRAY_SIZE 1024

int main() {
  OMPVV_TEST_OFFLOADING;
  OMPVV_INFOMSG("test target_teams_distribute_is_device_ptr");

  int a[ARRAY_SIZE];
  int b[ARRAY_SIZE];
  int *c = (int *)omp_target_alloc(ARRAY_SIZE * sizeof(int), omp_get_default_device());
  int errors = 0;

  if (!c) {
    OMPVV_WARNING("Test was unable to allocate memory on device.  Test could not procede.");
    OMPVV_REPORT_AND_RETURN(errors);
  } else {
    for (int x = 0; x < ARRAY_SIZE; ++x) {
      a[x] = 1;
      b[x] = x;
    }

#pragma omp target teams distribute is_device_ptr(c) map(tofrom: a[0:ARRAY_SIZE]) map(to: b[0:ARRAY_SIZE])
    for (int x = 0; x < ARRAY_SIZE; ++x) {
      c[x] = b[x] * b[x];
      a[x] += c[x] + b[x];
    }

    for (int x = 0; x < ARRAY_SIZE; ++x) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, (a[x] != 1 + b[x] + b[x] * b[x]));
      if (a[x] != 1 + b[x] + b[x] * b[x]) {
	break;
      }
    }

    omp_target_free (c, omp_get_default_device());

    OMPVV_REPORT_AND_RETURN(errors);
  }
}
