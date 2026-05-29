//===---- test_dispatch_novariants.c -----------------------------------------===//
// 
// OpenMP API Version 5.1
//
// Uses dispatch construct as context for variant directive. Uses novariants clause
// which can determine whether the variant function will be used.
// When novariants is true, then the variant cannot be used and vice versa. 
//
// Inspired by "OpenMP 5.1 Features: The Dispatch Construct" video:
// https://www.youtube.com/watch?v=ruugaX95gIs
// 
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ompvv.h"
#include <stdbool.h>

#define N 1024

int arr[N]; // implicit map array 
int errors;
int i = 0;

void add_two(int *arr);

#pragma omp declare variant(add_two) match(construct={dispatch})
void add(int *arr){
   for (int i = 0; i < N; i++){ // Base function adds 1 to array values
      arr[i] = i+1;
   }
}

void add_two(int *arr){
   for (int i = 0; i < N; i++){
      arr[i] = i+2; // Variant function adds 2 to array values
   }
}

int test_wrapper() { 
   errors = 0;
   int err_ar[2] = {0, 0};
   bool novariant_arg;
   add(arr);
   for(i = 0; i < N; i++){
      OMPVV_TEST_AND_SET(errors, arr[i] != i+1);
   } 
   OMPVV_ERROR_IF(errors > 0, "Base function is not working properly");
   novariant_arg = true;
   #pragma omp dispatch novariants(novariant_arg)
      add(arr);
   
   for(i = 0; i < N; i++){
      OMPVV_TEST_AND_SET(err_ar[0], (arr[i] != i + 1));
   }
   errors += err_ar[0];
   OMPVV_ERROR_IF(errors > 0, "Dispatch w/ novariants true is not working properly");

   novariant_arg = false;
   #pragma omp dispatch novariants(novariant_arg)
      add(arr);

   for(i = 0; i < N; i++){
      OMPVV_TEST_AND_SET(err_ar[1], (arr[i] != i + 1) && (arr[i] != i + 2));
   }
   // OMP 5.1, Pg54:6 -Whether the dispatch construct is added to the construct
   // set is implementation defined.

   errors += err_ar[1];
   OMPVV_INFOMSG_IF(errors > 0 || arr[0] == 1,
                   "Dispatch is either not working or was not considered"
                   " by the implementation as part of the context selector.");
   return errors;
}

int main () {
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
   OMPVV_REPORT_AND_RETURN(errors);
}  
