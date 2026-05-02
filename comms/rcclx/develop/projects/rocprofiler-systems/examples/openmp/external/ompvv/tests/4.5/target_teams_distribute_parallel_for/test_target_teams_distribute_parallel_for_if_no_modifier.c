//===---- test_target_teams_distribute_parallel_for_if_no_modifier.c --------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// In this test we want to try to check if the if clause is working 
// when used with the combined construct target teams distribute parallel for
// and no modifier is specified.
// To do this we check if offloading is working, if it is not, it won't be
// possible for us to tell if the test passed or not, since it depends on 
// offloading capabilities. 
//
// The if has a directive-name-modifier that specifies to which directive the 
// if applies to (either both directives, to the target or to the parallel). 
// we create three test, one for no directive, another one for the target directive
// and another one for the parallel directive 
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024

int checkPreconditions() {
  // We test if offloading is enable, and if 
  // the number of threads is not 1. Having 
  // the number of threads equal to 1 is legal, but 
  // we won't be able to test if the 'if' clause is 
  // affecting this or not

  // Testing for offloading
  int isOffloading = 0;
  int i;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test if on the parallel and not the target");

  // Testing for number of threads
  int init_num_threads_dev[N], init_num_threads_host[N];
  // init_num_threads_dev and init_num_threads_host arrays initialization
  for (i = 0; i < N; i++) {
    init_num_threads_dev[i] = 0;
    init_num_threads_host[i] = 0;
  }

  // Get the init_num_threads for host and device. If it is 1, then we 
  // cannot test the if parallel 
  // See section 2.5.1 of the ref manual
#pragma omp target teams distribute parallel for num_threads(OMPVV_NUM_THREADS_DEVICE)
  for (i = 0; i < N; i++) {
    init_num_threads_dev[i] = omp_get_num_threads();
  }

#pragma omp parallel for num_threads(OMPVV_NUM_THREADS_DEVICE)
  for (i = 0; i < N; i++) {
    init_num_threads_host[i] = omp_get_num_threads();
  }
  
  // We check that not all the values are 1. This would make the test of if parallel
  // not possible. 
  int raiseWarningDevice = 0, raiseWarningHost = 0;
  for (i = 0; i < N; i++) {
    if (init_num_threads_dev[i] > 1 ) {
      raiseWarningDevice +=  1;
    }
    if ( init_num_threads_host[i] > 1) {
      raiseWarningHost += 1;
    }
  }
  OMPVV_WARNING_IF(raiseWarningDevice == 0, "Initial number of threads in device was 1. It is not possible to test the if for the parallel directive");
  OMPVV_WARNING_IF(raiseWarningHost == 0, "Initial number of threads in host was 1. It is not possible to test the if for the parallel directive");

  return isOffloading;
}

int test_target_teams_distribute_if_no_modifier() {
  OMPVV_INFOMSG("test_target_teams_distribute_if_no_modifier");

  int isOffloading = checkPreconditions();

  int a[N];
  int warning[N] ; // num_threads = 1 is not technically an error
  int attempt = 0;
  int errors = 0;
  int i;

  for (i = 0; i < N; i++) {
    a[i] = 1;
    warning[i] = 0;
  }

  // We iterate NUM_ATTEMPTS times. When the iteration value is >= ATTEMPT_THRESHOLD the 
  // execution of the target region should happen in the devie, and the number of threads
  // is expected to be greater than 1 (although if it is not, this is not an error, just warning)
  // If the iteration variable attempt  < ATTEMPT_THRESHOLD, the if condition evaluates to false, 
  // and hence the execution should occur in the host with a number of threads equal to 1.
  // We check that when we are executing on the host, the value of the iteration variable is 
  // greather than ATTEMPT_THRESHOLD, and we make sure the number of threads is 1. 
  for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute parallel for if(attempt >= ATTEMPT_THRESHOLD)\
    map(tofrom: a, warning) num_threads(OMPVV_NUM_THREADS_DEVICE)
    for (i = 0; i < N; i++) {
      if (omp_is_initial_device()) {
        // if(false): We should execute in the host 
        // and the number of threads is expected 
        // to be 1
        a[i] += (omp_get_num_threads() > 1) ? 10 : 0; // This +10 should not happena
        a[i] += (attempt >= ATTEMPT_THRESHOLD) ? 10 : 0; // This +10 should not happena
      } else {
        a[i] += 1;
        warning[i] += (omp_get_num_threads() == 1 ? 1 : 0); // We cannot say that this is an error but we can raise a warning
      }
    }
  }

  int raiseWarning = 0;
  
  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, a[i] != 1 + (NUM_ATTEMPTS - ATTEMPT_THRESHOLD));
    if (warning[i] != 0) {
      raiseWarning = 1;
    }
  }
  OMPVV_WARNING_IF(raiseWarning != 0, "The number of threads was 1 even though we expected it to be more than 1. Not a compliance error in the specs");

  OMPVV_ERROR_IF(errors, "error in if(no-modifier). Possible causes are: the number of threads was greater than 1 for if(false), the test executed in the host for if(true), or the test executed in the device for if(false)");

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if_no_modifier());

  OMPVV_REPORT_AND_RETURN(errors);
}
