//===-------------- test_target_teams_loop_is_device_ptr.c----------------===//
//
// OpenMP API Version 5.0 Nov 2018
// The tests in this file tests 'teams loop' construct coupled with is_device_ptr
// & map() clause
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

/**
  This is a basic test to demonstrate is_device_ptr
  clause with target teams loop construct.
*/
int testIsDevPtrBasic() {
  int *a_h = NULL;
  int *a_d = NULL;
  int errors = 0;

  a_h = (int *) malloc(N * sizeof(int));
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_h == NULL);
  a_d = (int *) omp_target_alloc(N * sizeof(int),
                omp_get_default_device());
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_d == NULL);

#pragma omp target teams loop is_device_ptr(a_d) map(from: a_h[0:N])
  for (int i = 0; i < N; ++i) {
    a_d[i] = i;
    a_h[i] = a_d[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, a_h[i] != i);
  }
  omp_target_free(a_d, omp_get_default_device());
  free(a_h);
  return errors;
}

/**
  This is a basic test to demonstrate is_device_ptr
  clause with 2 target teams blocks. The first block populates
  data and the second block uses the data.
*/
int testIsDevPtrOn2DifferentTasks() {
  int *a_h = NULL;
  int *a_d = NULL;
  int errors = 0;

  a_h = (int *) malloc(N * sizeof(int));
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_h == NULL);
  a_d = (int *) omp_target_alloc(N * sizeof(int),
                omp_get_default_device());
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_d == NULL);

#pragma omp target teams loop is_device_ptr(a_d)
  for (int i = 0; i < N; ++i) {
    a_d[i] = i;
  }

#pragma omp target teams loop is_device_ptr(a_d) map(from: a_h[0:N])
  for (int i = 0; i < N; ++i) {
    a_h[i] = a_d[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, a_h[i] != i);
  }
  omp_target_free(a_d, omp_get_default_device());
  free(a_h);
  return errors;
}

/**
  This is a basic test to demonstrate is_device_ptr
  clause with 2 target teams blocks using multiple device pointers.
*/
int testIsDevPtrOn2DifferentPointers() {
  int *a_h = NULL;
  int *a_d1 = NULL, *a_d2 = NULL;
  int errors = 0;

  a_h = (int *) malloc(N * sizeof(int));
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_h == NULL);
  a_d1 = (int *) omp_target_alloc(N * sizeof(int),
                omp_get_default_device());
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_d1 == NULL);
  a_d2 = (int *) omp_target_alloc(N * sizeof(int),
                omp_get_default_device());
  OMPVV_TEST_AND_SET_VERBOSE(errors, a_d2 == NULL);

#pragma omp target teams loop is_device_ptr(a_d1, a_d2)
  for (int i = 0; i < N; ++i) {
    a_d1[i] = 2*i;
    a_d2[i] = 2*i + 1;
  }

#pragma omp target teams loop is_device_ptr(a_d1, a_d2) map(from: a_h[0:N])
  for (int i = 0; i < N; ++i) {
    a_h[i] = a_d1[i] + a_d2[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, a_h[i] != (4*i + 1));
  }
  omp_target_free(a_d1, omp_get_default_device());
  omp_target_free(a_d2, omp_get_default_device());
  free(a_h);
  return errors;
}

int main(void) {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testIsDevPtrBasic());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testIsDevPtrOn2DifferentTasks());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testIsDevPtrOn2DifferentPointers());
  OMPVV_REPORT_AND_RETURN(errors);
}
