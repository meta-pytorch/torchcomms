//--------------test_task_threadset_ancestor_THREADS_RESERVE.c----------------//
// OpenMP API Version 6.0 November 2024
// Pg. 901, line 30
// Pg. 588, line 22
// ***********
// DIRECTIVE:task
// CLAUSE:threadset
// ***********
// This test checks the functionality of the threadset clause with the task
// directive, while also checking for the correct use of the
// omp_ancestor_is_free_agent() routine. The example referenced here is "5.10
// Free Agent Threads" which is taken from the 6.0 examples document. It checks
// if the task is executed by a free-agent thread when threadset(omp_pool) is
// used, and that the result of the Fibonacci calculation is correct when both
// omp_pool and omp_team is used.
//----------------------------------------------------------------------------//

#include "ompvv.h"
#include <omp.h>

// not too large
#define N 5

int ancestor_count = 0;

int fib_seq(int n) {
  if (n < 2)
    return n;
  return fib_seq(n - 1) + fib_seq(n - 2);
}

int fib(int n, int use_pool) {
  int i, j;

  if (use_pool) {
    for (int k = 0; k <= omp_get_level(); ++k) {
      if (omp_ancestor_is_free_agent(k)) {
        #pragma omp atomic
        ancestor_count++;
      }
    }
  }
  if (n < 2)
    return n;
  if (use_pool) {
    #pragma omp task shared(i) threadset(omp_pool)
    i = fib(n - 1, use_pool);
    #pragma omp task shared(j) threadset(omp_pool)
    j = fib(n - 2, use_pool);
    #pragma omp taskwait
  } else {
    #pragma omp task shared(i) threadset(omp_team)
    i = fib(n - 1, use_pool);
    #pragma omp task shared(j) threadset(omp_team)
    j = fib(n - 2, use_pool);
    #pragma omp taskwait
  }

  return (i + j);
}

int task_work(int n, int use_pool) {
  int errors = 0;
  ancestor_count = 0;
  int result;

  omp_set_num_threads(OMPVV_NUM_THREADS_HOST);
  #pragma omp parallel
  #pragma omp single
  {
    result = fib(n, use_pool);
  }
  if (use_pool) {
    OMPVV_WARNING_IF(
        ancestor_count == 0,
        "no ancestor free-agent threads in region"); // ancestor_count should be
                                                     // greater than 0 if a
                                                     // free-agent ancestor
                                                     // thread executed the task
  }
  OMPVV_TEST_AND_SET(errors, result != fib_seq(n));
  return errors;
}

int test_task_threadset() {
  int errors = 0;
  int n = N;

  // Test task with threadset(omp_pool)
  errors += task_work(n, 1);

  // Test task with threadset omp_team
  errors += task_work(n, 0);

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET(errors, test_task_threadset() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
