!===--- declare_target_module.F90 ------------------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the declare target construct within a module, used in
! a target region. It checks the result of a simple offloaded addition.
! This test was provided by the LLNL FGPU repository, derived from the
! OpenMP 4.5 examples document. Find the repo at https://github.com/LLNL/FGPU/.
! Thanks to David Richards and Aaron Black for providing this test.
!
!===------------------------------------------------------------------------===//


#include "ompvv.F90"

#define N 1024

MODULE module_declare
CONTAINS
  SUBROUTINE test_declare(x)
    INTEGER:: x
    !$omp declare target
    x = x + 1
  END SUBROUTINE test_declare
END MODULE module_declare

MODULE params
  INTEGER:: THRESHOLD = N - 1
END MODULE params

PROGRAM declare_target_module
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  USE params
  USE module_declare
  implicit none
  INTEGER:: x

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_SHARED_ENVIRONMENT

  x = N

  !$omp target if(x > THRESHOLD) map(tofrom: x)
  CALL test_declare(x)
  !$omp end target

  OMPVV_TEST_VERBOSE(x .ne. N + 1)

  OMPVV_REPORT_AND_RETURN()
END PROGRAM declare_target_module
