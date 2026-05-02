//===--- test_omp_get_device_num.c -------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Test of omp_get_device_num() on both the host device and offloading
// device. When called on host device, omp_get_device_num() will return the 
// same value as omp_get_initial_device(). 
//
////===-----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1028

int errors;
int original_device_num;
int target_device_num; 
int a[N];
int b[N];
int c[N];


int test_omp_get_dev_num(void) {

   original_device_num = omp_get_device_num();

   OMPVV_ERROR_IF(original_device_num != omp_get_initial_device(), "omp_get_device_num() does not equal omp_get_initial_device()");
 
   OMPVV_TEST_AND_SET_VERBOSE(errors, original_device_num != omp_get_initial_device());

   for (int i = 0; i < N; i++) {
      a[i] = 0;
      b[i] = 5;
      c[i] = i;
   }
   
   #pragma omp target map(from: target_device_num) map(to: b, c) map(tofrom: a)
   { 
      target_device_num = omp_get_device_num();
      for (int i = 0; i < N; i++) {
         a[i] = b[i] + c[i];
      }
   }

   OMPVV_WARNING_IF(target_device_num == original_device_num, "omp_get_device_num() returned the same device number as host, cannot guarantee that target region properly offloaded to device");
   
   for (int i = 0; i < N; i++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != i + 5);
   }

   target_device_num = 99;

   for (int i = 0; i <= omp_get_num_devices(); i++) {
     #pragma omp target map(from: target_device_num) device(i)
     {
       target_device_num = omp_get_device_num();
     }

     OMPVV_TEST_AND_SET_VERBOSE(errors, target_device_num != i);
   }

   return errors;
}

int main () {

   errors = 0;
  
   OMPVV_TEST_OFFLOADING;

   OMPVV_TEST_AND_SET_VERBOSE(errors, test_omp_get_dev_num());

   OMPVV_REPORT_AND_RETURN(errors);
}
