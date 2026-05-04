//===--- test_target_update_to_present.c ----------------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  This test checks behavior of the target update clause when the specified motion-modifier  
//  is present. Tests 1. A corresponding list item and an original list item exist for each 
//  list item in a to or from clause. If the corresponding list item is not present in the
//  device data environment and the present modifier is not specified in the clause then no
//  assignment occurs to or from the original list item. Also tests 2. Otherwise, each
//  corresponding list item in the device data environment has an original list item in the
//  current task's data environment. 
//
////===--------------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024


int test_target_update_to_present() {
   
   int errors = 0;

   struct test_struct {
     int s; 
     int S[N]; 
   }; 

   int scalar_var; // scalar 
   int A[N]; // aggregate 
   struct test_struct new_struct; // aggregate
   int *ptr; // scalar, pointer 
   int isOffloadingOn = 0;

   OMPVV_TEST_AND_SET_OFFLOADING(isOffloadingOn);

   scalar_var = 1; 
   A[0] = 0; A[50] = 50;
   new_struct.s = 10; new_struct.S[0] = 10; new_struct.S[1] = 10;
   ptr = &A[0]; 
   ptr[50] = 50; ptr[51] = 51;
  
   // Tests OpenMP 5.1 Specification pp. 207 lines 2-4
   #pragma omp target update to(scalar_var, A, new_struct) 
   if (isOffloadingOn) {
      // Skip this test if target offloading is not enabled (running on the host).
      if (omp_target_is_present(&scalar_var, omp_get_default_device()))
        errors++;
      if (omp_target_is_present(&A, omp_get_default_device()))
        errors++;
      if (omp_target_is_present(&new_struct, omp_get_default_device()))
        errors++;
   }

   // Tests OpenMP 5.1 Specification pp. 207 lines 5-6
   #pragma omp target enter data map(alloc: scalar_var, A, new_struct)
   #pragma omp target update to(present: scalar_var, A, new_struct) 
   if (!omp_target_is_present(&scalar_var, omp_get_default_device()))
     errors++;
   if (!omp_target_is_present(&A, omp_get_default_device()))
     errors++;
   if (!omp_target_is_present(&new_struct, omp_get_default_device()))
     errors++;
   #pragma omp target map(tofrom: errors) defaultmap(none) map(to: scalar_var, A, new_struct)
   {     
        if(scalar_var != 1){errors++;}
        if(A[0] != 0 || A[50] != 50){errors++;}
        if(A[50] != 50 || A[51] != 51){errors++;}
        if(new_struct.s != 10){errors++;}
        if(new_struct.S[0] != 10){errors++;}
   }
   #pragma omp target exit data map(release: scalar_var, A, new_struct)
   
   return errors;
}

int main() {
   int errors = 0;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_update_to_present() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}           
