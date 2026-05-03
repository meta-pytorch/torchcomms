//===---- test_nested_declare_target.c -------------------------------------===//
//
// OpenMP API Version 5.0
//
// The declare target directive specifies that variables, functions(C,C++ and Fortran),
// and subroutines (Fortran) are mapped to a device. The declaration-definition-seq
// defined by a declare target directive and an end declare target directive may contain
// declare target directives.
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int errors = 0;
int a[N], b[N], c[N];

#pragma omp declare target
#pragma omp declare target link(a,b,c)
#pragma omp end declare target

void update() {
  for (int i = 0; i < N; i++) {
    a[i] += 1;
    b[i] += 2;
    c[i] += 3;
  }
}

#pragma omp declare target
#pragma omp declare target to(update)
#pragma omp end declare target

int test_declare_target_device_type_any() {

  #pragma omp target map(tofrom: a, b, c)
  {
    update();
  }


  for (int i = 0; i < N; i++) { //check array values on host
    if ( a[i] != i + 1 || b[i] != i + 3 || c[i] != i + 5) {
      errors++;
    }
  }

  //on host
  update();

  for (int i = 0; i < N; i++) { //check array values on host
    if ( a[i] != i + 2 || b[i] != i + 5 || c[i] != i + 8) {
      errors++;
    }
  }

  return errors;
}

int main () {

  //initalize arrays on host
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i + 1;
    c[i] = i + 2;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_device_type_any());

  OMPVV_REPORT_AND_RETURN(errors);
}
