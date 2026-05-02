//===--- test_task_affinity.c ----------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This is a test of the affinity clause on a task construct. The affinity
// clause indicates to the compiler that the task should execute physically
// near to the memory location of the list items in the clause. This test
// checks that the affinity clause can be used in the appropriate context
// of a task construct but cannot guarantee that the compiler provides any
// exact semantics for the clause.
//
////===----------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_task_affinity() {
  OMPVV_INFOMSG("test_task_affinity");
  int errors = 0;
  int* A;
  int* B;

  A = (int*) malloc(sizeof(int)*N);
  for (int i = 0; i < N; i++) {
    A[i] = 0;
  }

#pragma omp task depend(out: B) shared(B) affinity(A[0:N])
  {
    B = (int*) malloc(sizeof(int)*N);
    for (int i = 0; i < N; i++) {
      B[i] = A[i];
    }
  }

#pragma omp task depend(in: B) shared(B)
  {
    for (int i = 0; i < N; i++) {
      B[i] = i*2;
    }
  }

#pragma omp taskwait

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, B[i] != i*2);
    OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != 0);
  }

  free (A);
  free (B);

  return errors;
}

int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_affinity());

  OMPVV_REPORT_AND_RETURN(errors);
}
