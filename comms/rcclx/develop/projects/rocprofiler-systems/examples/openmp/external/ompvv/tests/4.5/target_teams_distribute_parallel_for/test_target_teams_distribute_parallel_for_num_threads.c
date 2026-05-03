//===---- test_target_teams_distribute_parallel_for_num_threads.c -----------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// Test to check the num_threads clause. This clause changes the upper limit of 
// the number of threads inside the parallel region, when used with the combined
// construct target teams distribute parallel for.
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define N 1024

int test_target_teams_distribute_parallel_for_num_threads() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_num_threads");
 
  int tested_num_threads[] = {1, 10, 100, 10000};
  int num_threads[N]; // num_threads = 1 is not technically an error
  int errors = 0;
  int raiseWarningOneThread = 0;
  int i, nt;


  // Trying multiple num_threads values from 1 to a large number
  for (nt = 0; nt < 4; nt++) {
  
    OMPVV_INFOMSG("Testing for num_threads(%d)", tested_num_threads[nt]);
    // Initializing the num_threads array
    for (i = 0; i < N; i++) {
      num_threads[i] = -1;
    }
#pragma omp target teams distribute parallel for\
        map(tofrom: num_threads) num_threads(tested_num_threads[nt])
    for (i = 0; i < N; i++) {
      num_threads[i] = omp_get_num_threads();
    }

    int raiseWarningDifNum = 0;
    int prevNumThreads = -1;
    
    for (i = 0; i < N; i++) {
      OMPVV_INFOMSG_IF(prevNumThreads != num_threads[i], " %d threads reported", num_threads[i]);
      prevNumThreads = num_threads[i];
      // If the number of threads is larger than the specified, this is an error
      OMPVV_TEST_AND_SET(errors, num_threads[i] <=0 || num_threads[i] > tested_num_threads[nt]);
      if (tested_num_threads[nt] != num_threads[i]) {
        raiseWarningDifNum = 1;
      }
      if (tested_num_threads[nt] != 1 && num_threads[i] == 1) {
        raiseWarningOneThread++;
      }
    }
    OMPVV_WARNING_IF(raiseWarningDifNum != 0 , "When testing num_threads(%d), the actual number of threads was different. This is not a compliance error with the specs", tested_num_threads[nt]);  
  }

  OMPVV_WARNING_IF(raiseWarningOneThread == 4*N, "The number of threads was always one, regardless of the num_threads clause. This is not a compliance error in the specs");  
  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_num_threads());

  OMPVV_REPORT_AND_RETURN(errors);
}
