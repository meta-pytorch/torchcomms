#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

/**
  This is a basic test to demonstrate how reduction clause is used
  with target teams loop construct.
*/
int testTargetTeamsReduction() {
  int a[N], total = 0;
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = i;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N]) map(tofrom: total) reduction(+:total)
  for (int i = 0; i < N; i++) {
    total = total + a[i];
  }
  // Validate
  int sum_host = 0;
  for (int i = 0; i < N; i++) {
    sum_host = sum_host + a[i];
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, total != sum_host);
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsReduction());
  OMPVV_REPORT_AND_RETURN(errors);
}
