//===------ test_declare_target_to_extended_list.c  ----------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks the second valid format of declare target directive with the 
// to clause. The extended list allows for mappable variables and function names 
// to be listed. If a list item of a to clause is a variable then the original 
// variable is mapped to a corresponding variable in the device data environment 
// of all devices as if it had appeared in a map clause with the map-type to on 
// the implicit target data construct for each device.
// Updates using the enclosed function are made inside the target region and
// results are verified on the host.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int aint = 10;

#pragma omp declare target to(aint)

void compute_array(int a[N], int b[N], int c[N]) {
  for (int i = 0; i < N; i++) {
    a[i] = b[i]*c[i] + aint * i;
  }
  return;
}
 
#pragma omp declare target to(compute_array)

int test_declare_target() {

  OMPVV_INFOMSG("test_declare_target_to_extended_list");

  int errors = 0;
  int x[N];
  int y[N];
  int z[N];

  for (int i = 0; i < N; i++) {
    x[i] = 0;
    y[i] = 1;
    z[i] = i;
  }

#pragma omp target map(from: x) map(to:y, z)
  {
    compute_array(x, y, z);
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != (y[i] * z[i] + 10 * i));
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
