//===--- test_target_teams_distribute_lastprivate.c--------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the lastprivate clause to indicate that the privatized value
// that is passed as the parameter should also be returned with the value that
// results from the thread that runs the last iteration of the for loop in the
// target teams distribute directive.  The clause can be used with both scalar
// and array data types and both situations are tested.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024

int main() {
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  int a[SIZE];
  int b[SIZE];
  int c[SIZE];
  int privatized = 0;
  int privatized_array[2];
  int errors = 0;

  for (int x = 0; x < SIZE; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = 0;
  }


#pragma omp target data map(to: a[0:SIZE], b[0:SIZE]) map(tofrom: c[0:SIZE])
  {
#pragma omp target teams distribute lastprivate(privatized) map(alloc: a[0:SIZE], b[0:SIZE], c[0:SIZE]) \
  defaultmap(tofrom:scalar)
    for (int x = 0; x < SIZE; ++x) {
      privatized = a[x] - b[x];
      c[x] = privatized + b[x];
    }
  }

  for (int x = 0; x < SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[x] - a[x] != 0);
    if (c[x] - a[x] != 0) {
      break;
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, privatized != a[SIZE - 1] - b[SIZE - 1]);

  for (int x = 0; x < SIZE; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = x % 10;
  }

#pragma omp target data map(to: a[0:SIZE], b[0:SIZE], c[0:SIZE]) map(tofrom: privatized_array[0:2])
  {
#pragma omp target teams distribute lastprivate(privatized_array) map(alloc: a[0:SIZE], b[0:SIZE], c[0:SIZE])
    for (int x = 0; x < SIZE; ++x) {
      privatized_array[0] = a[x] + b[x] + c[x];
      privatized_array[1] = (a[x] + b[x]) * c[x];
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, privatized_array[0] != (a[SIZE - 1] + b[SIZE - 1] + c[SIZE - 1]));
  OMPVV_TEST_AND_SET_VERBOSE(errors, privatized_array[1] != ((a[SIZE - 1] + b[SIZE - 1]) * c[SIZE - 1]));
  OMPVV_REPORT_AND_RETURN(errors);
}
