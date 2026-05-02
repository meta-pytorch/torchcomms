//===-------- test_task_target.c - test task with target offload  ------------===//
// 
// OpenMP API Version 4.5 Nov 2015
//
// This test checks target regions inside of tasks. With offloading support 
// the tasks are executed on the device. The results do not change if the 
// test is executed on the host. Output will indicate if the test passed and
// where (host/device).
//
////===----------------------------------------------------------------------===//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define N 1000

int test_task_target() {
  int a[N], b[N], c[N];
  int errors = 0, i;

#pragma omp task shared(a) private(i)
#pragma omp target map(from: a)
#pragma omp parallel for
  for (i = 0; i < N; i++)
    a[i] = i;
   
#pragma omp task shared(b) private(i)
#pragma omp target map(from: b)
#pragma omp parallel for
  for (i = 0; i < N; i++)
    b[i] = 10;

#pragma omp taskwait

#pragma omp task shared(c) private(i)
#pragma omp target map(from: c) map(to:a,b)
#pragma omp parallel for
  for (i = 0; i < N; i++)
    c[i] = a[i] + b[i];

#pragma omp taskwait

  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, (c[i] != i + 10));
  }
  return(errors);
}

int main() {

  OMPVV_TEST_OFFLOADING;

  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_target());

  OMPVV_REPORT_AND_RETURN(errors);
}
