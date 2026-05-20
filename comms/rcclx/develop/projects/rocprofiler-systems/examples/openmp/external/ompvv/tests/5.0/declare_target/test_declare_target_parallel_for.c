//===------ test_declare_target_parallel_for.c ----------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Test of a function that is declared target and contains a parallel for
// construct. The function performs simple array operations on the device
// which are then checked for correctness on the host. The number of
// threads used to run the parallel for is also checked.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp declare target
int parallel_for_fun(int a[N], int b[N], int c[N]) {
  int num_threads = -1;
#pragma omp parallel for num_threads(OMPVV_NUM_THREADS_DEVICE) shared(num_threads)
  for (int i = 0; i < N; i++) {
    a[i] = b[i]*c[i];
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }
  return num_threads;
}
#pragma omp end declare target

int test_declare_target_parallel_for() {
  OMPVV_INFOMSG("test_declare_target_parallel_for");
  int errors = 0;
  int x[N];
  int y[N];
  int z[N];
  int num_threads = -1;

  for (int i = 0; i < N; i++) {
    x[i] = 0;
    y[i] = 1;
    z[i] = i;
  }

#pragma omp target map(tofrom: x, y, z, num_threads)
  {
    num_threads = parallel_for_fun(x, y, z);
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != y[i]*z[i]);
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);
  OMPVV_ERROR_IF(num_threads < 1, "Device returned invalid number of threads.");
  OMPVV_WARNING_IF(num_threads == 1, "Device ran target function with parallel for using one thread, so parallelism cannot be guaranteed.");

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_parallel_for() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
