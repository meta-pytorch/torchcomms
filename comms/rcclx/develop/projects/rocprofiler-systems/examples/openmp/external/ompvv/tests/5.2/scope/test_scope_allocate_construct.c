//--------------- test_scope_allocate_construct.c ----------------------------//
//
// OpenMP API Version 5.2 Nov 2021
//
// This test checks that the scope directive with the allocate clause is
// working correctly.
// Note: restrictions for the allocate clause require that a private clause
// be present when allocate is used with scope.
// The allocator() argument to the allocate clause selects one of the
// predefined allocators which are listed under Table 6.3 on pg. 174
// in the specifications.
//
// In this test, the clause is used to allocate memory to an array of integers,
// which is then written to. Each section of the array is subsequently read
// to check if the correct integer values are present.
//----------------------------------------------------------------------------//

#include "ompvv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

int test_allocate() {
  int errors = 0;
  int arr[N];

  #pragma omp target parallel map(tofrom : errors)
  #pragma omp scope private(arr) allocate(allocator(omp_low_lat_mem_alloc) : arr)
  {
    int err = 0;
    for (int i = 0; i < N; i++) {
      arr[i] = i;
    }
    for (int j = 0; j < N; j++) {
      if (arr[j] != j)
        err++;
    }
    #pragma omp atomic update
    errors += err;
  }
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
