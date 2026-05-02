//===--- test_target_simd_order_concurrent.c -----------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks for support of the order(concurrent) clause on a target simd construct.
// When an order(concurrent) clause is present on a simd construct, all of the same 
// restrictions from having a loop construct with an order(concurrent) are also applied.
//
////===-------------------------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <omp.h> 
#include "ompvv.h" 

#define N 1024

int test_simd_order_concurrent () {
   int errors = 0;
   int i;
   int b[N], c[N];

   struct new_struct {
      int a[N];
   };

   struct new_struct struct_t;
   
   for (i = 0; i < N; i++ ) {
      struct_t.a[i] = i;
      b[i] = i + 5;
      c[i] = 0;
   }

   #pragma omp target simd order(concurrent)
      for (i = 0; i < N; i++) {
         c[i] = struct_t.a[i] * b[i];
      }

   for (i = 0; i < N; i++) {
      OMPVV_TEST_AND_SET(errors, c[i] != (struct_t.a[i] * b[i]));
   }

   return errors;
}


int main () { 
   int errors = 0;

   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_simd_order_concurrent());
   OMPVV_REPORT_AND_RETURN(errors);
}
