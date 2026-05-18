//===--- test_target_enter_data_global_array.c ------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//  
// This is a test of the target enter data construct with global arrays.
// The 'to' map-type-modifier is specified on the map clause.
//
//===------------------------------------------------------------------------===//

#include "ompvv.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Test for OpenMP 4.5 target enter data with global arrays.

int n=10;
int A[10]={10,10,10,10,10,10,10,10,10,10},B[10];


int main (){

 int i;
 int errors = 0;
 
 OMPVV_TEST_OFFLOADING;

#pragma omp target enter data map(to: A[:n])
#pragma omp target map(from: B[:n])
{

 for (i = 0; i < n; i++)
    B[i] = A[i];
}

 for (i = 0; i < n; i++)
    if (B[i] != 10){
     errors += 1;
   }

#pragma omp target exit data map(release: A[:n])

  OMPVV_REPORT_AND_RETURN(errors);
}
