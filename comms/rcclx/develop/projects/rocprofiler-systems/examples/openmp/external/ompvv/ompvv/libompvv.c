//===------ libompvv.c --------------- OMPVV STATIC LIBRARY ---------------===//
//
// Static lib file for OMP Validation and verification test suite.
//
//===----------------------------------------------------------------------===//

#include "omp.h"
#include "libompvv.h"

int offload_test() {
  int a = 0;

#pragma omp target map(tofrom: a)
  {
    a = 1;
  }

  return a;
}
