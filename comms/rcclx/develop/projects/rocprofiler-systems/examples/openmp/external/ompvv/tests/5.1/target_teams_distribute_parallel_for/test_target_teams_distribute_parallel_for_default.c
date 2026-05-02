//-----------------test_target_teams_distribute_parallel_for_default.c-----
//
// OpenMP API Version 5.1 November 2020
//
// This test case targets to test default clause along with
// target teams distribute parallel for construct
// --------------------------------------------------------------



#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"


int DefaultFirstPrivate() {
  int ErrCount = 0;
  int Arr[32];
  for (int i = 0; i < 32; ++i) {
    Arr[i] = i;
  }

#pragma omp target teams distribute parallel for num_teams(2) thread_limit(10)\
        default(firstprivate) shared(ErrCount) map(tofrom: ErrCount)
  for (int i = 0; i < 32; ++i) {
    if (Arr[i] != i) {
      #pragma omp atomic
      ErrCount += 1;
    }
  }
  return ErrCount;
}

int DefaultPrivate() {
  int ErrCount = 0;
  int CONST = 123;
  int Arr[32];
  for (int i = 0; i < 32; ++i) {
    Arr[i] = i;
  }

#pragma omp target teams distribute parallel for num_teams(2) thread_limit(10)\
        default(private) shared(ErrCount) firstprivate(Arr) map(to: Arr[0:32]) map(tofrom: ErrCount)
  for (int i = 0; i < 32; ++i) {
    CONST = 10;
    Arr[i] += CONST;
    if (Arr[i] != (i + 10)) {
      #pragma omp atomic
      ErrCount += 1;
    }
  }
  if (CONST != 123) {
    ErrCount++;
  }
  return ErrCount;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, DefaultFirstPrivate() != 0);
  OMPVV_TEST_AND_SET_VERBOSE(errors, DefaultPrivate() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
