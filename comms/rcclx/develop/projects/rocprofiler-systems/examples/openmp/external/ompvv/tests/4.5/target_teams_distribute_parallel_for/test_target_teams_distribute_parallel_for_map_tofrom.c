//===---- test_target_teams_distribute_parallel_for_map_tofrom.c - combined consutrct -===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// Testing the map clause with the tofrom map-modifier, for the combined construct
// target teams distribute parallel for.
// Scalar mapping has to be divided between to and from. Otherwise there will be 
// data races between the threads that are writting the scalar and those that are reading
// from it
//
//===-----------------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define N 2000

int test_target_teams_distribute_parallel_for_map_tofrom() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_map_tofrom");
  
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int scalar_to = 50; //to avoid datarace on the scalar due to read and write in the loop
  int scalar_from = 50;
  int errors = 0;
  int i, j, dev;


  // variables initialization
  scalar_to = 50;
  scalar_from = 50;
  for (i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = i;
    c[i] = 2*i;
    d[i] = 0;
  }

    // Tests
#pragma omp target teams distribute parallel for map(tofrom: a, b, c, d, scalar_to, scalar_from)
  for (j = 0; j < N; ++j) {
    d[j] += c[j] * (a[j] + b[j] + scalar_to);
    a[j] = 10;
    b[j] = 11;
    c[j] = 12;
#pragma omp atomic write
    scalar_from = 13; // This is to avoid data races on a single scalar
  }

  // Checking the results
  OMPVV_TEST_AND_SET(errors, scalar_from != 13);
  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, a[i] != 10);
    OMPVV_TEST_AND_SET(errors, b[i] != 11);
    OMPVV_TEST_AND_SET(errors, c[i] != 12);
    OMPVV_TEST_AND_SET(errors, d[i] != (1 + i + 50) * 2*i);
  }
  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_map_tofrom());

  OMPVV_REPORT_AND_RETURN(errors);
}
