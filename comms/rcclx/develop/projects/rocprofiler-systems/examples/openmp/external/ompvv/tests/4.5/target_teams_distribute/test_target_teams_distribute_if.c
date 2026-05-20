//===--- test_target_teams_distribute_if.c-----------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the if clause to specify whether the target teams distribute
// directve should execute in the device or the hosts through the if clause of the 
// taget directive. The test uses omp_is_initial_device through the different 
// threads, assigning an expected value to a matrix according to 1) offloading is 
// enabled and working, 2) the if clause evaluates to true and the code executes in
// the offloading device, and 3) the if clause evaluates to false and the code executes
// in the offloading devices. 
//
// If not operating on a device, the test has a minimal test
// of the basic use of the if clause with both a true and a false parameter.
// However, the execution is identical to that of host operation in both cases.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100

int test_target_teams_distribute_if() {
  OMPVV_INFOMSG("test_target_teams_distribute_if");

  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test if");

  int a[N];
  int errors = 0;
  int attempt = 0;
  int i;

  for (int x = 0; x < N; ++x) {
    a[x] = 1;
  }

  for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute if(attempt >= ATTEMPT_THRESHOLD) map(tofrom: a)
    for (i = 0; i < N; ++i) {
      if (attempt >= ATTEMPT_THRESHOLD) {
	a[i] += (isOffloading && omp_is_initial_device() ? 10 : 0); // false -> running on device or not offloading -> add 0
      } else {
	a[i] += (omp_is_initial_device() ? 1 : 100);                // true -> running on host -> add 1
      }
    }
  }

  for (i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET(errors, a[i] != (1 + ATTEMPT_THRESHOLD));
  }

  if (errors) {
    int sum = 0;
    for (i = 0; i < N; ++i) {
      sum += a[i];
    }
    if (sum == N*(100*ATTEMPT_THRESHOLD + 1)) {
      OMPVV_ERROR("Error in if. The execution was expected to occur on the host, but it occurred on the device.");
    } else if (sum == N*(ATTEMPT_THRESHOLD + 10*(NUM_ATTEMPTS - ATTEMPT_THRESHOLD) + 1)) {
      OMPVV_ERROR("Error in if. The execution was expected to occur on the device, but it occurred on the host.");
    } else {
      OMPVV_ERROR("Error in if. The execution occurred inconsistently on the host or on the device.");
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if());
  
  OMPVV_REPORT_AND_RETURN(errors);
}
