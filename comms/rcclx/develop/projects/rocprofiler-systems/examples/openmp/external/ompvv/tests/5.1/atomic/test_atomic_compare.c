//===--- test_atomic_compare.c ---===//
//
// OpenMP API Version 5.1 Nov 2020
//
// Adapted from OpenMP example video https://www.youtube.com/watch?v=iS6IG7nzCSo
// Creates an array with random numbers, and uses atomic compare to find the max,
// testing against non-parallel maximum.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 100

int test_atomic_compare() {
  OMPVV_INFOMSG("test_atomic_compare");

  int arr[N];
  int errors = 0;
  int pmax = 0, smax = 0;

   for(int i=0; i<N; i++){
      arr[i] = rand()%1000;
   }
   for(int i = 0; i<N; i++){ // Sets max through non-parallel methods
      if(arr[i] > smax){
         smax = arr[i];
      }
   }
   #pragma omp parallel for shared(pmax)// Sets max using parallel for loop, using atomic to ensure max is correct
   for(int i = 0; i<N; i++){
      #pragma omp atomic compare
      if(arr[i] > pmax){
         pmax = arr[i];
      }
   }
   OMPVV_TEST_AND_SET(errors, pmax != smax);
   return errors;
}


int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_compare());
  OMPVV_REPORT_AND_RETURN(errors);
}
