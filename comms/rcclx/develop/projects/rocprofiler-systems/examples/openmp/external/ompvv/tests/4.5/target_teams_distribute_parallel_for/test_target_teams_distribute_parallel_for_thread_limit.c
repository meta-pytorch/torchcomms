//===---- tests_target_teams_distribute_parallel_for_thread_limit.c ---------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// Test to check the thread_limit clause. This clause changes the upper limit of 
// the number of threads inside each of the contention groups created in the teams
// region. This upper limit is different to the num_threads 
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define N 1024

int test_target_teams_distribute_parallel_for_thread_limit() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_threads_limit");

  int tested_num_threads[] = {1, 10, 100, 10000};
  int tested_thread_limit[] = {1, 10, 100, 10000};
  int num_threads[N];
  int thread_limit[N];
  int errors = 0;
  int i, nt, tl;

  // Trying multiple num_threads and thread_limits
  // values from 1 to a large number. The number of threads
  // should never be larger than the thread limit
  for (nt = 0; nt < 4; nt++) {
    for (tl = 0; tl < 4; tl++) {
      OMPVV_INFOMSG("Testing thread_limit(%d) num_threads(%d) clauses", tested_thread_limit[tl], tested_num_threads[nt]);
      // Initializing the num_threads and thread_limit array
      for (i = 0; i < N; i++) {
        num_threads[i] = -1;
        thread_limit[i] = -1;
      }

#pragma omp target teams distribute parallel for map(tofrom: num_threads) \
        num_threads(tested_num_threads[nt]) thread_limit(tested_thread_limit[tl])
      for (i = 0; i < N; i++) {
        num_threads[i] = omp_get_num_threads();
        thread_limit[i] = omp_get_thread_limit();
      }
   
      int prevThreadLimit = -1;
      for (i = 0; i < N; i++) {
        OMPVV_INFOMSG_IF(prevThreadLimit != thread_limit[i], "  reported thread limit = %d", thread_limit[i]);
        prevThreadLimit = thread_limit[i];
  
        // If the thread limit is not the one specified or the actual number 
        // of threads is greater than the thread limit, then it is an error
        OMPVV_TEST_AND_SET_VERBOSE(errors, (thread_limit[i] > tested_thread_limit[tl]) || (thread_limit[i] <= 0));
        OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads[i] > tested_thread_limit[tl]);
        // For completeness, if the actual number of threads is grater than the 
        // selected number of threads, then it is an error
        OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads[i] > tested_num_threads[nt]);
      }
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_thread_limit());

  OMPVV_REPORT_AND_RETURN(errors);
}
