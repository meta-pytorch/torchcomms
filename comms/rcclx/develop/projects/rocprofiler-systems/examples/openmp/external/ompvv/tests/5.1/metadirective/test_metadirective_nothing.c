//===--------------------- test_metadirective_nothing.c ---------------------===//
//
// OpenMP API Version 5.1 Nov 2020
// 
// Test for nothing directive within metadirectives. Runs a variety of
// metadirectives that check if the nothing directive is properly rendered.
// Primarily tests based on the fact that no matter what 'when' clause is 
// rendered it should result in nothing, and thus no additional pragma should
// be created. Thus, the threads should remain unchanged through this process
// and the compiler should handle it properly. Handles both device and host
// based tests.
//
////===---------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
#define THREAD_LIMIT 32

int metadirectiveOnDevice() {
   int errors = 0;
   int A[N];
   int thread_limit_target = THREAD_LIMIT;
   int thread_limit_teams = 0;

   for (int i = 0; i < N; i++) {
      A[i] = 0;
   }
   #pragma omp target map(from: thread_limit_target) thread_limit(thread_limit_target)
   {
      thread_limit_target = omp_get_thread_limit();
   }

   #pragma omp target map(tofrom: A, thread_limit_teams) thread_limit(thread_limit_target)
   {
      // We expect at least one of these when conditons to eval to true, thus having the nothing directive utilized
      #pragma omp metadirective \
         when( device={kind(nohost)}: nothing ) \
         when( device={arch("nvptx")}: nothing) \
         when( implementation={vendor(amd)}: nothing ) \
         default( teams distribute parallel for thread_limit(thread_limit_target+1) )
            for (int i = 0; i < N; i++) {
               A[i] += omp_in_parallel();
               if( i == 0 ) {
                 thread_limit_teams = omp_get_thread_limit();
               }
            }
   }

   for (int i = 0; i < N; i++) {
      OMPVV_TEST_AND_SET(errors, A[i] != 0);
   }
   OMPVV_TEST_AND_SET(errors, thread_limit_target != thread_limit_teams);

   OMPVV_INFOMSG("Test ran with a number of available devices greater than 0");
   OMPVV_INFOMSG_IF(thread_limit_target == thread_limit_teams, "Test recognized device was of arch/vendor/kind nvidia, amd, or nohost");
   OMPVV_WARNING_IF(A[0] == 1, "Test could not recognize if device was of arch/vendor/kind nvidia, amd or, nohost, even though there are devices available.");

   return errors;
}

int metadirectiveOnHost() {
  int errors = 0;
  int A[N];
  int thread_limit_initial = THREAD_LIMIT;
  int thread_limit_teams = 0;

  for (int i = 0; i < N; i++) {
     A[i] = 0;
  }

  thread_limit_initial = omp_get_thread_limit();
  // We expect all of these when statements to eval to false, causing body of code to run using 'nothing' as the default pragma
  #pragma omp metadirective \
     when( device={kind(nohost)}: teams distribute parallel for thread_limit(thread_limit_initial+1) ) \
     when( device={arch("nvptx")}: teams distribute parallel for thread_limit(thread_limit_initial+1) ) \
     when( implementation={vendor(amd)}: teams distribute parallel for thread_limit(thread_limit_initial+1) ) \
     default( nothing )
        for (int i = 0; i < N; i++) {
           A[i] += omp_in_parallel();
           if( i == 0 ) {
              thread_limit_teams = omp_get_thread_limit();
           }
        }

  OMPVV_WARNING_IF(A[0] == 1, "Even though no devices were available the test recognized kind/arch equal to nohost or nvptx or amd");
  
  for (int i = 0; i < N; i++) {
     OMPVV_TEST_AND_SET(errors, A[i] != 0);
  }
  OMPVV_TEST_AND_SET(errors, thread_limit_initial != thread_limit_teams);

  return errors;
} 

int main () {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;

  if (omp_get_num_devices() > 0) {
    errors = metadirectiveOnDevice();
  } else {
    errors = metadirectiveOnHost();
  }

  OMPVV_REPORT_AND_RETURN(errors);

  return 0;
}
