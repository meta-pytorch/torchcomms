#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

/**
  This is a basic test to demonstrate device clause used
  with "omp target teams loop".
*/
int main() {
  int a[N];
  int errors = 0, total_dev = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 0;  // Set to 0th device
  }
  total_dev = omp_get_num_devices();
  for (int dev_num = 0; dev_num < total_dev; dev_num++) {
  // Execute on target
  #pragma omp target teams loop map(tofrom: a[0:N]) device(dev_num)
    for (int i = 0; i < N; i++) {
      a[i] = a[i] + 1;
    }
  }
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != total_dev);
  }
  OMPVV_REPORT_AND_RETURN(errors);
}
