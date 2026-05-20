!===--- test_declare_target_nested_functions.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Test of declare target on a host function that calls another host
! function. According to 5.0 specification, the inner function, while
! not explicitly declared on the target, should be treated as if it was.
! The declared functions increment the passed-in value and return it. 
! They are tested in a simple target region and the result is mapped out
! and checked.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_declare_target_nested_functions

    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    implicit none
    INTEGER :: errors
    LOGICAL :: is_offloading
    
    !$omp declare target to(outer_fn)
    
    errors = 0 
     
    OMPVV_TEST_AND_SET_OFFLOADING(is_offloading)

    OMPVV_TEST_VERBOSE(test_declared_functions() .ne. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS  
    INTEGER FUNCTION inner_fn(a, on_device)
      INTEGER:: a
      LOGICAL :: on_device

      if (omp_is_initial_device()) on_device = .false.
      inner_fn = 1 + a
    END FUNCTION inner_fn

    INTEGER FUNCTION outer_fn(a, on_device)
      INTEGER:: a
      LOGICAL :: on_device
      
      if (omp_is_initial_device()) on_device = .false.
      outer_fn = 1 + inner_fn(a, on_device)
    END FUNCTION outer_fn
    
    INTEGER FUNCTION test_declared_functions() 
      INTEGER:: outcome, errors
      LOGICAL :: on_device
      errors = 0
      outcome = 0
      on_device = .false.

      !$omp target map (tofrom: outcome, on_device)
      on_device = .true.
      outcome = outer_fn(outcome, on_device)
      !$omp end target

      OMPVV_TEST_AND_SET_VERBOSE(errors, on_device .neqv. is_offloading)
      OMPVV_TEST_AND_SET_VERBOSE(errors, outcome .ne. 2)

      test_declared_functions = errors
    END FUNCTION test_declared_functions

END PROGRAM test_declare_target_nested_functions 
