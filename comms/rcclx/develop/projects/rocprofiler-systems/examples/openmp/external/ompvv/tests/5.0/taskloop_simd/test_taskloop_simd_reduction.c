//===--- test_taskloop_simd_reduction.c -------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the taskloop simd combined directive with the reduction clause specified.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_taskloop_reduction() {

   int errors = 0;
   int a[N];
   int b[N]; 
   int sum = 0;
   int num_threads = -1;
   int real_sum = 0;

   for (int i = 0; i < N; i++) {
      a[i] = 5;
      b[i] = i * 2; 
   }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST) shared(a, b, num_threads, sum) 
{
   #pragma omp single
   #pragma omp taskloop simd reduction(+:sum)
   for (int i = 0; i < N; i++) {
      #pragma omp atomic
      sum += a[i]*b[i]; 
   }
   num_threads = omp_get_num_threads();

   #pragma omp single
   #pragma omp taskloop simd reduction(+:sum)
   for (int i = 0; i < N; i++) {
      #pragma omp atomic
      sum++;
   }

}

   real_sum += N;

   for (int i = 0; i < N; i++) {
      real_sum += a[i]*b[i];
   }

   OMPVV_TEST_AND_SET_VERBOSE(errors, sum != real_sum);

   OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of taskloop can't be guaranteed.");
   OMPVV_ERROR_IF(num_threads < 1, "Test returned an invalid number of threads.");
   OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);


   return errors;            
}


int main() {

   int errors = 0;

   OMPVV_TEST_AND_SET_VERBOSE(errors, test_taskloop_reduction());

   OMPVV_REPORT_AND_RETURN(errors);

}
