//===--test_target_taskloop_shared.c ----------------------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// Test uses a value within a taskloop & an atomic construct to update the value of
// s_val. The shared clause ensures that the s_val will be shared between threads,
// and therefore should be equal to N After the taskloop region.
//
////===--------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"

#define N 1024

int taskloop_shared() {

  int errors = 0;
  int s_val=0;

  #pragma omp target map(tofrom: s_val)
  {
    #pragma omp parallel
    {
      #pragma omp single
      #pragma omp taskloop shared(s_val)
      for (int i = 0; i < N; ++i){
        #pragma omp atomic update
        ++s_val; 
      }
    }
  }
  OMPVV_ERROR_IF(s_val != N, "Value of s_val should be %i, received %i", N, s_val);
  OMPVV_TEST_AND_SET(errors, s_val != N);
  return errors;
}

int main() {
    int errors = 0;
    OMPVV_TEST_OFFLOADING;
    OMPVV_TEST_AND_SET(errors, taskloop_shared());
    OMPVV_REPORT_AND_RETURN(errors);
}
