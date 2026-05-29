//===---- test_omp_in_explicit_task.c ----------------------------------------------===//
// 
// OpenMP API Version 5.2 Nov 2021
//
// Uses omp_in_explicit_task() runtime call to ensure it is 1 when in an explicit
// task and 0 otherwise.
// 
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ompvv.h"
#include <stdbool.h>

#define N 256

int errors;

int test_wrapper() { 
    errors = 0;
    int A[N];
    for(int i = 0; i < N; i++){
        A[i] = 0;
    }
    #pragma omp task shared(A)// creates EXPLICIT task, omp_in_explicit_task = 1
    {
        for(int i = 0; i < N; i++){
            A[i] = omp_in_explicit_task();
        }
    }
    for(int i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, A[i] == 0);
    }
    OMPVV_ERROR_IF(A[rand()%N] == 0, "omp_in_explicit_task() did not return correct value when called inside an explicit task");
    
    for(int i = 0; i < N; i++){
        A[i] = 1;
        #pragma omp parallel for // creates IMPLICIT tasks, omp_in_explicit_task = 0
        for(int i = 0; i < N; i++){
            A[i] = omp_in_explicit_task();
        }
    }
    OMPVV_ERROR_IF(A[rand()%N] != 0, "omp_in_explicit_task() did not return correct value when called inside an implicit task");
    
    for(int i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, A[i] != 0);
    }
    return errors;
}

int main () {
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
   OMPVV_REPORT_AND_RETURN(errors);
}  
