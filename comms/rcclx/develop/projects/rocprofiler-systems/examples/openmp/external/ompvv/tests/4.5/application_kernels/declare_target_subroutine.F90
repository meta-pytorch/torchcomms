!===--- declare_target_subroutine.F90 --------------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses a declare target construct to within a subroutine, called
! on the device. It takes in an array argument and checks that the array
! is summed correctly. This test was provided by the LLNL FGPU repository,
! https://github.com/LLNL/FGPU/, derived from the OpenMP 4.5 examples.
! Thanks to David Richards and Aaron Black for providing this test.
!
! Note that the use of an explicit interface differs from that implicit
! interface shown in OpenMP 4.5 examples on page 116 (declare_target.2.f90).
! This is because the test_declare function uses a deferred-shape array
! argument. Thanks to Tobias Burnus for correcting this.
!
!===------------------------------------------------------------------------===//


#include "ompvv.F90"

#define N 1024

PROGRAM declare_target_subroutine
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER,POINTER:: a(:)
  INTEGER:: x, sum

  INTERFACE
     SUBROUTINE test_declare(a, sum)
       INTEGER,POINTER,INTENT(in) :: a(:)
       INTEGER,INTENT(inout):: sum
     END SUBROUTINE test_declare
  END INTERFACE

  sum = 0

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_SHARED_ENVIRONMENT

  ALLOCATE(a(N))
  DO x = 1, N
     a(x) = 1
  END DO

  !$omp target map(tofrom: sum) map(to: a)
  CALL test_declare(a, sum)
  !$omp end target

  OMPVV_TEST_VERBOSE(sum .ne. N)

  DEALLOCATE(a)
  OMPVV_REPORT_AND_RETURN()
END PROGRAM declare_target_subroutine

SUBROUTINE test_declare(a, sum)
  INTEGER,POINTER,INTENT(in) :: a(:)
  INTEGER,INTENT(inout):: sum
  INTEGER:: x
  !$omp declare target
  DO x = 1, N
     sum = sum + a(x)
  END DO
END SUBROUTINE test_declare
