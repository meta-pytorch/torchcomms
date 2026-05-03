//===-- qmcpack_target_static_lib.c ---------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks that a target region can be compiled as a static
// library. The library function offload_test contains a target region
// which changes the value of a scalar on the device and maps it back.
// This value is then checked by this driver code.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#include "libompvv.h"

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, offload_test() != 1);

  OMPVV_REPORT_AND_RETURN(errors);
}
