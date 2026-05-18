#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

/**
  This is a basic test to demonstrate private clause used
  with "omp target teams loop".
*/
int main() {
  int a[N], b[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = a[i];
  }
  // Execute on target
#pragma omp target teams loop map(tofrom: b[0:N]) private(a)
  for (int i = 0; i < N; i++) {
    a[i] = 2*i + 1;
    b[i] = a[i];
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, b[i] != (2*i + 1));
    OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != (2*i));
  }
  OMPVV_REPORT_AND_RETURN(errors);
}
