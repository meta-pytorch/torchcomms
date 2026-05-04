//===---- test_dispatch_is_device_ptr.c -----------------------------------------===//
// 
// OpenMP API Version 5.1
//
// Uses dispatch construct as context for variant directive. Uses is_device_ptr
// clause to ensure that the list item is in the device region.
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
int *arr;

void add_dev(int *arr);

#pragma omp declare variant(add_dev) match(construct={dispatch}) 
void add(int *arr){
    #pragma omp target parallel for is_device_ptr(arr)
    for (int i = 0; i < N; i++){ // Base function adds 2 to array values
        arr[i] = arr[i]+2;
    }
}

void add_dev(int *arr){
    #pragma omp target is_device_ptr(arr)
    for (int i = 0; i < N; i++){
        arr[i] = arr[i]+4; // Variant function adds 4 to array values
    }
}

int test_wrapper() { 
    int t;
    int called_add = 0;
    errors = 0;
    t = omp_get_default_device();
    arr = (int *)omp_target_alloc( sizeof(int)*N, t);
    if(arr == NULL){
        OMPVV_ERROR("Can't properly create device pointer");
        return 1;
    }
    #pragma omp target is_device_ptr(arr)
    {
        #pragma omp parallel for
        for(int i = 0; i < N; i++){
            arr[i] = i;
        }
    }
    #pragma omp dispatch is_device_ptr(arr)
        add(arr);
    
    #pragma omp target map(tofrom: errors, called_add) is_device_ptr(arr)
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, (arr[i] != i+4) && (arr[i] != i+2) );
        called_add += arr[i] == i + 2;
    }
    // Note that it is implementation deﬁned in OpenMP 5.1 (and 5.2) whether
    // the dispatch construct is added to the construct set. Only since
    // OpenMP Technical Review 12 (TR12; second 6.0 preview) it must be added;
    // in that case, 'add_dev' is called and called_add == 0.
    OMPVV_INFOMSG_IF(errors > 0 || called_add > 0,
                   "Dispatch is either not working or was not considered"
                   " by the implementation as part of the context selector.");
    omp_target_free (arr, t);
    return errors;
}

int main () {
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
   OMPVV_REPORT_AND_RETURN(errors);
}  
