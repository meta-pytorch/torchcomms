//===--- test_target_enter_exit_data_map_global_array.c ---------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test is in two parts. First, the test checks that mapping to on enter
// followed by mapping from on exit works, by modifying the data on the
// device. Then, the delete clause is tested by making sure that deleting
// an array mapped to the device resets its reference count, meaning that
// modifications made on the host are remapped back in when another map(to)
// is encountered.
//
////===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <omp.h>

#define N 10

int A[N] = {0};
int B[N] = {0};

int test_tofrom() {
  int errors = 0;

  for (int i = 0; i < N; ++i) {
    A[i] = 0;
  }

#pragma omp target enter data map(to: A)

#pragma omp target
  {
    for (int i = 0; i < N; i++) {
      A[i] = N;
    }
  }

#pragma omp target exit data map(from: A)

  for (int i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != N);
  }

  return errors;
}

int test_delete() {
  int errors = 0;

  for (int i = 0; i < N; ++i) {
    A[i] = N;
  }

#pragma omp target data map(tofrom: A) map(from: B)
  {
#pragma omp target exit data map(delete: A)
    for (int i = 0; i < N; ++i) {
      A[i] = 0;
    }
#pragma omp target map(to: A)   // if the delete does not work, this map will not happen.
    {
      for (int i = 0; i < N; ++i) {
        B[i] = A[i];
      }
    }
  }

  for (int i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, B[i] != 0);
  }

  return errors;
}

int main () {
  int errors = 0;

  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_SHARED_ENVIRONMENT;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_tofrom() != 0);
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_delete() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
