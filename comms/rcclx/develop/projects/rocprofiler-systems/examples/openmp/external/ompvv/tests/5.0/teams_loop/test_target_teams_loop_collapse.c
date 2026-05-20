#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define DIM 32
#define N (DIM*DIM)

/**
  This is a basic test to demonstrate collapse clause used
  with "omp target teams loop".
*/
int main() {
  int a[N], b[N], c[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    c[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a, b) map(from: c) collapse(2)
  for (int i = 0; i < DIM; i++) {
    for (int j = (i*DIM); j < (i*DIM + DIM); j++) {
      c[j] = a[j] + b[j];
    }
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
  }
  OMPVV_REPORT_AND_RETURN(errors);
}
