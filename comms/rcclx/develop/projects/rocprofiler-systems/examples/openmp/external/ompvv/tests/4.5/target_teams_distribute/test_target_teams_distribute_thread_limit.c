//===--- test_target_teams_distribute_thread_limit.c--------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the thread_limit clause on a target teams distribute directive to
// indicate a requested number of threads to execute the teams distribute region.
// The specifications indicate that the number of threads that are given can be any
// number that is equal to or less than the indicated value. 
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_target_teams_distribute_thread_limit(){
  int num_threads = -9;
  int errors = 0;
  
  
  #pragma omp target teams distribute thread_limit(4) map(from: num_threads)
    for (int x = 0; x < N; ++x) {
      if (omp_get_team_num() == 0) {
        #pragma omp parallel
        {
        if (omp_get_thread_num() == 0)
          num_threads = omp_get_num_threads();
        }
      }
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads > 4);

  return errors;

}

int main(){
  int errors = 0;
  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_thread_limit() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}

