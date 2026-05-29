#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <time.h>

#define N 1024

/**
  This is a basic test to demonstrate nowait clause used
  with "omp target teams loop".
*/
int main() {
  int *a, *b, *c;
  a = (int*) malloc(N*sizeof(int));
  b = (int*) malloc(N*sizeof(int));
  c = (int*) malloc(N*sizeof(int));
  if ((a == NULL) || (b == NULL) || (c == NULL)) {
    OMPVV_REPORT_AND_RETURN(1);
  }
  int errors = 0;
  srand(time(NULL));
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = (rand() % 1024) + 1;
    b[i] = (rand() % 1024) + 1;
    c[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N]) nowait
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
#pragma omp taskwait
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
  }
  free(a);
  free(b);
  free(c);
  OMPVV_REPORT_AND_RETURN(errors);
}
