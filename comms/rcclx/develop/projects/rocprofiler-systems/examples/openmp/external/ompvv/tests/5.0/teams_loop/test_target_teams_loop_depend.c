//===---------------test_target_teams_loop_depend.c---------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// The tests in this file tests 'teams loop' construct coupled with map()
// and depend clause
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
#define INC 7
#define MUL 11

/**
  This is a basic test to demonstrate how depend clause in various
  combinations.
*/
int testDependAddToMulSync() {
  int a[N], b[N], c[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 0;
    c[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(tofrom: a[0:N]) depend(out: a[0:N]) nowait
  for (int i = 0; i < N; i++) {
    a[i] += INC;
  }
  // Dependant on Addition
#pragma omp target teams loop map(tofrom: a[0:N], b[0:N]) depend(in: a[0:N])\
        depend(out: b[0:N]) nowait
  for (int i = 0; i < N; i++) {
    b[i] = a[i]*MUL;
  }
  // dependant on multiplication
#pragma omp target teams loop map(tofrom: b[0:N], c[0:N]) depend(in: b[0:N])
  for (int i = 0; i < N; i++) {
    c[i] = b[i];
  }
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != ((i + INC)*MUL));
  }
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDependAddToMulSync());
  OMPVV_REPORT_AND_RETURN(errors);
}
