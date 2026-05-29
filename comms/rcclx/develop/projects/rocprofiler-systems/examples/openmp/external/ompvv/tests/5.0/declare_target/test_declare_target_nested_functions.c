//===------ test_declare_target_nested_functions.c ------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Test of declare target on a host function that calls another host
// function. According to 5.0 specification, the inner function, while not
// explicitly declared on the target, should be treated as if it was. The
// declared functions increment the passed-in value and return it. They are
// tested in a simple target region and the result is mapped out and checked.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int inner_fn(int a) {
  return 1 + a;
}

int outer_fn(int a) {
  return 1 + inner_fn(a);
}

int test_declared_functions() {
  int result = 0, errors = 0;

#pragma omp target map(tofrom: result)
  {
    result = outer_fn(result);
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, result != 2)

    return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

#pragma omp declare target to(outer_fn)

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declared_functions() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
