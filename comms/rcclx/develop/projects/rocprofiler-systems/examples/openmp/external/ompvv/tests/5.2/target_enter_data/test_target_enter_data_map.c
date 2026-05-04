//===--- test_target_enter_data_map.c -----------------------------------===//
//
// OpenMP API Version 5.2 May 2022
//
// This is a test of the target enter data map construct with global arrays.
// The 'to' map-type-modifier is not specified on the map clause.
// The 'from' map-type-modifier is not specified on the map clause.
// Specification update 5.1 -> 5.2 target enter data has a default map type 
// that perfroms the same behavior as 'to' or 'from' map types
//
//===--------------------------------------------------------------------===//

#include "ompvv.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Test for OpenMP 5.2 target enter data with global arrays.

int main (){

  int N=10;
  int A[10]={10,10,10,10,10,10,10,10,10,10};
  int B[10];
  int D[10];
  int i = 0;
  int errors = 0;

  for (i = 0; i < N; i++) {
    B[i] = 50;
    D[i] = 0;
  }



  OMPVV_TEST_OFFLOADING;

  #pragma omp target enter data map(A[:N], B[:N], D[:N])
  //should default to map to
  #pragma omp target map(tofrom: errors)
  { 
    for (i = 0; i < N; i++){
      if(B[i] != 50){errors++;}
      if(A[i] != 10){errors++;}
      if(D[i] != 00){errors++;}
      
    }


    for (i = 0; i < N; i++) {
      D[i] = 70;
      B[i] = 120;
      A[i] = 1000;
    }
  }

  for (i = 0; i < N; i++){
    OMPVV_TEST_AND_SET_VERBOSE(errors, B[i] == 120)
    OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] == 1000)
    OMPVV_TEST_AND_SET_VERBOSE(errors, D[i] == 70)
  }




  #pragma omp target exit data map(B[:N], A[:N],D[:N])
  //should default to map from
  for (i = 0; i < N; i++){
    OMPVV_TEST_AND_SET_VERBOSE(errors, B[i] != 120)
    OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != 1000)
    OMPVV_TEST_AND_SET_VERBOSE(errors, D[i] != 70)
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
