//===---- test_dispatch_depend.c -----------------------------------------===//
// 
// OpenMP API Version 5.1
//
// Uses dispatch construct as context for variant directive. Uses depend clause
// which adds depend property to the interoperability requirement set.
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

#define N 1024

int arr[N]; 
int errors;
int i = 0;
int a = 0;

void add_a(int *arr);

#pragma omp declare variant(add_a) match(construct={dispatch})
void add(int *arr){
    for (int i = 0; i < N; i++){ // Base function adds 1 to array values
        arr[i] = i+1;
    }
    a = 3;
}

void add_a(int *arr){
    OMPVV_TEST_AND_SET(errors, a != 3);
    OMPVV_ERROR_IF(errors > 0, "Depend clause on dispatch directive not working properly");
    for (int i = 0; i < N; i++){
        arr[i] = i+a; // Variant function adds 3 to array values
    }
}

int test_wrapper() { 
    errors = 0;
    add(arr);
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != i+1);
    } 
    OMPVV_ERROR_IF(errors > 0, "Base function is not working properly");
    
    #pragma omp task
    {
    #pragma omp taskwait depend(out:a) nowait
    add(arr);
    #pragma omp dispatch depend(in: a)
    add(arr);
    }
    #pragma omp taskwait 

    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != i+1 && arr[i] != i+3);
    }

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
