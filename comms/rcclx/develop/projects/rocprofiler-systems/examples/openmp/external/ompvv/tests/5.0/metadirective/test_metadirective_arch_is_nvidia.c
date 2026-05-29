//===---test_metadirective_arch_is_nvidia.c --------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// 
// Test for metadirectives based on OpenMP 5.0 examples metadirective.1-3.c
//
////===---------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int metadirective1() {
   
   int v1[N], v2[N], v3[N];

   int target_device_num, host_device_num, default_device;
   int errors = 0;

   for(int i=0; i<N; i++) { 
      v1[i] = (i+1); 
      v2[i] = -(i+1); 
   }

   host_device_num = omp_get_initial_device();

   default_device = omp_get_default_device();

   #pragma omp target map(to:v1,v2) map(from:v3, target_device_num) device(default_device)
   {
      #pragma omp metadirective \
                   when(   device={arch("nvptx")}: teams distribute parallel for) \
                   default(                     parallel for)

         for (int i = 0; i < N; i++) {
	    #pragma omp atomic write
            target_device_num = omp_get_device_num();
            v3[i] = v1[i] * v2[i];
         }
   }
   
   OMPVV_TEST_AND_SET(errors, host_device_num == target_device_num);
   OMPVV_ERROR_IF(host_device_num == target_device_num, "Device number that executes target region is "
                                                        "the same as the device number on the host");
 
   for (int i = 0; i < N; i++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, v3[i] != v1[i] * v2[i]);
   }

   return errors;
}

int main () {
   
   int errors = 0;
   OMPVV_TEST_OFFLOADING;
 
   OMPVV_TEST_AND_SET_VERBOSE(errors, metadirective1());
  
   OMPVV_REPORT_AND_RETURN(errors);

}
