//===----  test_target_teams_distribute_parallel_for_reduction.c--===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// testing the combined construct target teams distribute parallel for
//
//===---------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N_Els 1024

int ReductionPlus() {
  int ErrCount = 0;
  int Arr[N_Els];
  for (int i = 0; i < N_Els; ++i) {
    Arr[i] = i + 1;
  }
  int Total = 0;

#pragma omp target teams distribute parallel for reduction(+:Total)
  for (int i = 0; i < N_Els; ++i) {
    Total += Arr[i];
  }
  if (Total != (N_Els * ((N_Els) + 1)/2)) {
    ErrCount++;
  }
  return ErrCount;
}

int ArrayReduction() {
  int ErrCount = 0;
  int Arr[N_Els][N_Els];
  for (int j = 0; j < N_Els; ++j) {
    for (int i = 0; i < N_Els; ++i) {
      Arr[j][i] = i + 1;
    }
  }
  int Total[N_Els];
  for (int i = 0; i < N_Els; ++i) {
    Total[i] = 0; 
  }
  int i, j;
#pragma omp target teams distribute parallel for reduction(+:Total[0:N_Els]) private(j)
  for (i = 0; i < N_Els; ++i) {
    for (j = 0; j < N_Els; ++j) {
      Total[i] += Arr[i][j];
    }
  }

  for (int i = 0; i < N_Els; ++i) {
    if (Total[i] != (N_Els * (N_Els + 1)/2)) {
      ErrCount++;
    }
  }
  return ErrCount;
}


int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, ReductionPlus() != 0);
  OMPVV_TEST_AND_SET_VERBOSE(errors, ArrayReduction() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
