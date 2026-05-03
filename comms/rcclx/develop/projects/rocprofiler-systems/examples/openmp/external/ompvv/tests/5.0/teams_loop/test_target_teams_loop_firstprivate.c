#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
/**
  This is a basic test to demonstrate firstprivate clause used
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
#pragma omp target teams loop map(tofrom: b[0:N]) firstprivate(a)
  for (int i = 0; i < N; i++) {
    b[i] = a[i] + 1;
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, b[i] != (2*i + 1));
  }
  OMPVV_REPORT_AND_RETURN(errors);
}
