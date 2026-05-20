//===--- test_atomic_hint.c -------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_atomic_hint_uncontended() {
  OMPVV_INFOMSG("test_atomic_hint_uncontended");
  int errors = 0, x = 0, num_threads = -1;

#pragma omp parallel num_threads(2) default(shared)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
#pragma omp atomic hint(omp_sync_hint_uncontended)
    x++;
  }

  OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of teams (less than zero)");
  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");

  OMPVV_TEST_AND_SET_VERBOSE(errors, x != num_threads);

  return errors;
}

int test_atomic_hint_contended_nonspec() {
  OMPVV_INFOMSG("test_atomic_hint_contended_nonspec");
  int errors = 0, x = 0, num_threads = -1;

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST) default(shared)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
#pragma omp atomic hint(omp_sync_hint_contended+omp_sync_hint_nonspeculative)
    x++;
  }

  OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of teams (less than zero)");
  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");

  OMPVV_TEST_AND_SET_VERBOSE(errors, x != num_threads);

  return errors;
}

int test_atomic_hint_speculative() {
  OMPVV_INFOMSG("test_atomic_hint_speculative");
  int errors = 0, num_threads = -1;
  int a[N];

  for (int i = 0; i < N; i++) {
    a[i] = 1;
  }

#pragma omp parallel for num_threads(OMPVV_NUM_THREADS_HOST) default(shared)
  for (int i = 0; i < N; i++) {
    if (i == 0) {
      num_threads = omp_get_num_threads();
#pragma omp atomic hint(omp_sync_hint_speculative)
      a[1] += 1;
    }
#pragma omp atomic hint(omp_sync_hint_speculative)
    a[i] += i;
  }

  OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of teams (less than zero)");
  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");


  for (int i = 0; i < N; i++) {
    if (i == 1) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != 3);
    } else {
      OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != i + 1);
    }
  }

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_hint_uncontended());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_hint_contended_nonspec());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_hint_speculative());

  OMPVV_REPORT_AND_RETURN(errors);
}
