//===---- reduction_separated_directives.cpp ------------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks the behavior of a reduction clause with separated target teams
// parallel for directives by comparing the results to those from atomic-protected
// writes without a reduction. This test comes from the aomp development repo, see
// https://github.com/ROCm-Developer-Tools/aomp/issues/51.
//
//===----------------------------------------------------------------------------------===//

#include <cstdio>
#include "omp.h"
#include "ompvv.h"

#define N 1024

int main()
{
  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_SHARED_ENVIRONMENT;

  int errors = 0;
  int counts_atomic = 0;
  int counts_reduction = 0;

#pragma omp target teams map(from: counts_atomic)
  {
    int counts_team = 0;
#pragma omp parallel
    {
#pragma omp for
      for (int i = 0; i < N; ++i)
#pragma omp atomic
        counts_team += 1;
    }
    if (omp_get_team_num() == 0) {
      counts_atomic = counts_team;
    }
  }

#pragma omp target teams map(from: counts_reduction)
  {
    int counts_team = 0;
#pragma omp parallel
    {
#pragma omp for reduction(+: counts_team)
      for (int i = 0; i < N; ++i)
        counts_team += 1;
    }
    if (omp_get_team_num() == 0) {
      counts_reduction = counts_team;
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, counts_atomic != N);
  OMPVV_TEST_AND_SET_VERBOSE(errors, counts_reduction != N);

  OMPVV_REPORT_AND_RETURN(errors);
}
