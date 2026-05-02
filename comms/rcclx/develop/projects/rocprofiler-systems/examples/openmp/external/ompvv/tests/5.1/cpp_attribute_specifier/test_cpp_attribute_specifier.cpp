//===--- test_cpp_attribute_specifier.cpp --------------------------------------------------------------===//
//
//  OpenMP API Version 5.1 NOV 2021
//  
//  C++ attribute specifiers allow for a different syntax for calling OMP directives.
//  This test simply uses a for loop, and tests multiple times that it is not running in parallel. Then,
//  using C++ OMP attribute specifiers, creates a parallel for loop and ensures that it is running in
//  parallel.
//
////===-------------------------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int errors, i;

int test_cpp_attribute_specifier() {
   omp_set_num_threads(4);
   for(int i = 0; i < N; i++){
       OMPVV_TEST_AND_SET_VERBOSE(errors, omp_in_parallel() == true);
   }
   [[ omp::sequence(directive(parallel), directive(for)) ]]
    for(int i = 0; i < N; i++){
       OMPVV_TEST_AND_SET_VERBOSE(errors, omp_in_parallel() != true);
   }
   return errors;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_cpp_attribute_specifier() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}           
