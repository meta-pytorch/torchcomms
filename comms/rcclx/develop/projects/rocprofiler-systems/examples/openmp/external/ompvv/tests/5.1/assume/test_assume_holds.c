//===--- test_assume_holds.c ------------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  The specification states that "When the holds clause appears on an assume
//  directive, the application guarantees that the listed expression evaluates
//  to true in the scope of the directive." In this test, we use "N" as a
//  constant variable that does not change in the scope of the assume directive.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

int errors, i, N;

int test_assume_holds() {
    N = 1024;
    int arr[N];
    #pragma omp assume holds(N == 1024)
    {
        for(i = 0; i < N; i++){
            arr[i] = i;
        }
        #pragma omp target parallel for map(tofrom: arr)
        for(i = 0; i < N; i++){
            arr[i] = arr[i]*2;
        }
    }
    // Test that no issues were caused from using assume directive
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != i*2);
    }
   return errors;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_assume_holds() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
