//===--- test_target_teams_distribute_collapse.c--------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the collapse clause and tests that for loops out of the scope
// of the collapsed loops are not parallelized.  This test tests using one and
// two collapsed loops.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define ARRAY_SIZE 128 //Array Size of 128 uses 16MB target memory and
                       //scales n^3 in test_collapse2()

int test_collapse1() {

  int * a_mem = (int *)malloc(ARRAY_SIZE*ARRAY_SIZE*sizeof(int));
  int * b_mem = (int *)malloc(ARRAY_SIZE*(ARRAY_SIZE+1)*sizeof(int));
  int (*a)[ARRAY_SIZE] = (int (*)[ARRAY_SIZE])a_mem;
  int (*b)[ARRAY_SIZE + 1] = (int (*)[ARRAY_SIZE+1])b_mem;
  int errors = 0;

  // a and b array initialization
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    b[x][0] = 0;
    for (int y = 0; y < ARRAY_SIZE; ++y) {
      a[x][y] = x + y;
      b[x][y+1] = 0;
    }
  }

#pragma omp target teams distribute map(to: a[0:ARRAY_SIZE][0:ARRAY_SIZE]) map(tofrom: b[0:ARRAY_SIZE][0:ARRAY_SIZE+1]) collapse(1) num_teams(OMPVV_NUM_TEAMS_DEVICE)
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    for (int y = 0; y < ARRAY_SIZE; ++y) {
      b[x][y + 1] = b[x][y] + a[x][y];
    }
  }

  int temp_total;
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    temp_total = 0;
    for (int y = 0; y < ARRAY_SIZE+1; ++y) {
      OMPVV_TEST_AND_SET(errors, ((temp_total - b[x][y]) != 0));
      if (y != ARRAY_SIZE) {
        temp_total = temp_total + a[x][y];
      }
    }
  }

  free (a_mem);
  free (b_mem);
  return errors;
}

int test_collapse2() {
  int * a_mem = (int *)malloc(ARRAY_SIZE*ARRAY_SIZE*ARRAY_SIZE*sizeof(int));
  int * b_mem = (int *)malloc(ARRAY_SIZE*ARRAY_SIZE*(ARRAY_SIZE+1)*sizeof(int));
  int (*a)[ARRAY_SIZE][ARRAY_SIZE] = (int (*)[ARRAY_SIZE][ARRAY_SIZE])a_mem;
  int (*b)[ARRAY_SIZE][ARRAY_SIZE + 1] = (int (*)[ARRAY_SIZE][ARRAY_SIZE+1])b_mem;
  int errors = 0;
  int num_teams = 0;

  // a and b array initialization
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    for (int y = 0; y < ARRAY_SIZE; ++y) {
      b[x][y][0] = 0;
      for (int z = 0; z < ARRAY_SIZE; ++z) {
        a[x][y][z] = x + y + z;
        b[x][y][z+1] = 0;
      }
    }
  }

#pragma omp target teams distribute map(to: a[0:ARRAY_SIZE][0:ARRAY_SIZE][0:ARRAY_SIZE]) map(tofrom: b[0:ARRAY_SIZE][0:ARRAY_SIZE][0:ARRAY_SIZE+1], num_teams) collapse(2) num_teams(OMPVV_NUM_TEAMS_DEVICE)
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    for (int y = 0; y < ARRAY_SIZE; ++y) {
      for (int z = 0; z < ARRAY_SIZE; ++z) {
        if (omp_get_team_num() == 0) {
          num_teams = omp_get_num_teams();
        }
        b[x][y][z + 1] = b[x][y][z] + a[x][y][z];
      }
    }
  }

  int temp_total;
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    for (int y = 0; y < ARRAY_SIZE; ++y) {
      temp_total = 0;
      for (int z = 0; z < ARRAY_SIZE + 1; ++z) {
        OMPVV_TEST_AND_SET(errors, ((temp_total - b[x][y][z]) != 0));
        if (z != ARRAY_SIZE) {
          temp_total = temp_total + a[x][y][z];
        }
      }
    }
  }

  if (num_teams == 1) {
    OMPVV_WARNING("Test operated with one team.  Parallelism of teams distribute can't be guaranteed.");
  }

  free (a_mem);
  free (b_mem);
  return errors;
}

int main() {
  
  //Check for offloading
  OMPVV_TEST_OFFLOADING;  

  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_collapse1() != 0);
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_collapse2() != 0);
  OMPVV_REPORT_AND_RETURN(errors);

}
