//===---- test_declare_variant_adjust_args.c -----------------------------------------===//
// 
// OpenMP API Version 5.1
//
// Uses declare variant directive with the adjust_args clause, which states that
// "For each adjust_args clause that is present on the selected variant the adjustment operation
// specified by adjust-op will be applied to each of the arguments specified in the clause
// before being passed to the selected variant." Clause also requires a
// match(construct={dispatch}) clause to be specified, as listed in restrictions.
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

int errors;
int i = 0;

void add_dispatch(int *arr);

#pragma omp declare variant(add_dispatch) match(construct={dispatch}) adjust_args(need_device_ptr:arr)
void add(int *arr){
    for (int i = 0; i < N; i++){ // Base function adds 1 to array values
        arr[i] = i+1;
    }
}

void add_dispatch(int *arr){
    if (arr == NULL){
      return;
    }
    #pragma omp target parallel for
    for (int i = 0; i < N; i++){
        arr[i] = i+3; // Variant function adds 3 to array values
    }
}

int test_wrapper() { 
    int *arr = (int*)malloc(sizeof(int)*N); // implicit map array 
    errors = 0;
    add(arr);
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i+1);
    } 
    OMPVV_ERROR_IF(errors > 0, "Base function is not working properly");

    #pragma omp target enter data map(to:arr[0:N])
    #pragma omp dispatch
        add(arr);  
        /* array should be converted to device ptr; i.e. spec states since it is not a device ptr, "the argument
        will be converted in the same manner that a use_device_ptr clause on a target data construct converts 
        its pointer list items into device pointers." */
    
    #pragma omp target parallel for map(tofrom: errors)
    for(i = 0; i < N; i++){
        if(arr[i] != i+3){
            errors++;
        }
    }
    #pragma omp target exit data map(delete:arr[0:N])
    free(arr);
    
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
