!===--- test_requires_unified_shared_memory.F90 ----------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test just uses requries unified_shared_memory in a simple target region
!  and it is intended to show that this would not break anything in the compiler
!
!===------------------------------------------------------------------------------===//

#define OMPVV_MODULE_REQUIRES_LINE !$omp requires unified_shared_memory
#include "ompvv.F90"

PROGRAM test_requires_unified_shared_memory
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none
   INTEGER:: errors, test_var
   LOGICAL:: isOffloading

!$omp requires unified_shared_memory

   OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)

   OMPVV_WARNING_IF(isOffloading .eqv. .false., "With no offloading, unified shared memory is guaranteed due to host execution")

   errors = 0

   test_var = 0

   OMPVV_INFOMSG("Unified shared memory testing")

!$omp target map(tofrom: test_var)

   test_var = test_var + 10

!$omp end target

   test_var = test_var + 10

   OMPVV_TEST_AND_SET_VERBOSE(errors, test_var .ne. 20)

   OMPVV_REPORT_AND_RETURN()

END PROGRAM test_requires_unified_shared_memory
      
   
