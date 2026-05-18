//===---- test_dispatch_device.c -----------------------------------------===//
// 
// OpenMP API Version 5.1
//
// Uses dispatch construct as context for variant directive. When the dispatch
// construct is reached, the variant function, add_two, should be used.
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

int errors;
int i = 0;
int arr[N];

void add_two(int *arr);

#pragma omp declare variant(add_two) match(construct={dispatch}) 
void add(int *arr){
    #pragma omp parallel for
    for (int i = 0; i < N; i++){ // Base function adds 1 to array values
      arr[i] = arr[i]+1;
    }
}

void add_two(int *arr){
    for (int i = 0; i < N; i++){
      arr[i] = arr[i]+2; // Variant function adds 2 to array values
    }
}

int test_wrapper() { 
    errors = 0;

    add(arr);
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != 1);
    } 
    OMPVV_ERROR_IF(errors > 0, "Base function is not working properly");

    #pragma omp dispatch
        add(arr);
   
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != 3 && arr[i] != 2);
    }
    OMPVV_INFOMSG_IF(errors > 0 || arr[0] == 2,
                   "Dispatch is either not working or was not considered"
                   " by the implementation as part of the context selector.");
    return errors;
}

int main () {
    for (int i = 0; i < N; i++){
      arr[i] = 0;
    }
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
   OMPVV_REPORT_AND_RETURN(errors);
} 
