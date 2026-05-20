//===---test_requires_unified_address.c -------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// 
// This test checks for support of unified_address clause on the requires
// directive.
//
/////===---------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_address

int unified_address() {

   int errors = 0;
   int i;
   int *mem_ptr = (int *)omp_target_alloc(N * sizeof(int), omp_get_default_device());
   int *mem_ptr2 = NULL;

   OMPVV_ERROR_IF(mem_ptr == NULL, "Target memory was not properly allocated");
   if (mem_ptr == NULL)
     return ++errors;
   
   #pragma omp target map(from:mem_ptr2) defaultmap(firstprivate) /* is_device_ptr(mem_ptr) - which is optional.  */
   {
      for (i = 0; i < N; i++) {
         mem_ptr[i] = i + 1;
      }
      mem_ptr2 = &mem_ptr[0] + 5;
   }

   /* Pointer arithmetic is permitted; assumes sizeof(int) is the same.  */
   mem_ptr2 += 4;
   
   #pragma omp target map(tofrom:errors) defaultmap(to) /* is_device_ptr(mem_ptr2) - which is optional.  */
   for (i = 0; i < N; i++) {
      if(mem_ptr2[i - 5 - 4] != i + 1) {
         errors++;
      }  
   }

   int *mem_ptr3 = (int*)malloc(N * sizeof(int));
   OMPVV_ERROR_IF(mem_ptr3 == NULL, "Host memory was not properly allocated");
   if (mem_ptr3 == NULL)
     return ++errors;

   if (0 == omp_target_memcpy(mem_ptr3, mem_ptr, N * sizeof(int), 0, 0,
                              omp_get_initial_device(), omp_get_default_device())) {
     for (i = 0; i < N; i++) {
       if (mem_ptr3[i] != i + 1) {
         errors++;
       }
     }
   } else {
     OMPVV_ERROR("omp_target_memcpy failed");
     errors++;
   }

   free (mem_ptr3);
   omp_target_free (mem_ptr, omp_get_default_device());

   return errors;
}

int main() {
  
  int errors = 0;

  OMPVV_TEST_OFFLOADING;
  OMPVV_TEST_AND_SET_VERBOSE(errors, unified_address());
  OMPVV_REPORT_AND_RETURN(errors);

}  
