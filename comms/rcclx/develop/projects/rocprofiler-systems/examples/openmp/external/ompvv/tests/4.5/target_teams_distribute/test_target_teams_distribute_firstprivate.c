//===--- test_target_teams_distribute_firstprivate.c-------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the firstprivate clause and tests it in two separate parts.
// The first test checks the privatization of the firstprivatized scalars and
// an array and the second tests the proper initialization of both
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int main() {
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int num_teams[N];
  int privatized_array[10];
  int privatized = 0;
  int ishost;
  int errors = 0;

  for (int x = 0; x < N; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = 2*x;
    d[x] = 0;
    num_teams[x] = -1;
  }

  for (int x = 0; x < 10; ++x) {
    privatized_array[x] = 0;
  }

  //Test privitization of data in firstprivate clause
#pragma omp target data map(from: d[0:N]) map(to: a[0:N], b[0:N], c[0:N])
  {
#pragma omp target teams distribute firstprivate(privatized_array, privatized) \
  map(alloc: a[0:N], b[0:N], c[0:N], d[0:N]) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    for (int x = 0; x < N; ++x) {
      num_teams[x] = omp_get_num_teams();
      for (int y = 0; y < a[x] + b[x]; ++y) {
        privatized++;
        for (int z = 0; z < 10; ++z) {
          privatized_array[z]++;
        }
      }
      d[x] = c[x] * privatized;
      for (int z = 0; z < 10; ++z) {
        d[x] += privatized_array[z];
      }
      privatized = 0;
      for (int z = 0; z < 10; ++z) {
        privatized_array[z] = 0;
      }
    }
  }

  for (int x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, d[x] != 10*(1 + x) + (1 + x)*2*x);
    if (d[x] != 10*(1 + x) + (1 + x)*2*x) {
      break;
    }
    OMPVV_WARNING_IF(num_teams[x] == 1, "Did not create enough teams to check for potential data races.");
  }

  privatized = 1;
  for (int x = 0; x < 10; ++x) {
    privatized_array[x] = x;
  }

  for (int x = 0; x < N; ++x) {
    num_teams[x] = -1;
  }

  //Test initialization of data in firstprivate clause
#pragma omp target data map(from: d[0:N]) map(to: a[0:N], b[0:N], c[0:N])
  {
#pragma omp target teams distribute firstprivate(privatized_array, privatized) \
  map(alloc: a[0:N], b[0:N], c[0:N], d[0:N]) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    for (int x = 0; x < N; ++x) {
      num_teams[x] = omp_get_num_teams();
      d[x] = a[x] + b[x] + c[x] + privatized_array[x%10] + privatized;
    }
  }

  int temp;
  for (int x = 0; x < N; ++x) {
    temp = x%10;
    OMPVV_TEST_AND_SET_VERBOSE(errors, d[x] != 2 + 3*x + temp);
    if (d[x] != 2 + 3*x + (x%10)) {
      break;
    }
    OMPVV_WARNING_IF(num_teams[x] == 1, "Did not create enough teams to check for potential data races.");
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
