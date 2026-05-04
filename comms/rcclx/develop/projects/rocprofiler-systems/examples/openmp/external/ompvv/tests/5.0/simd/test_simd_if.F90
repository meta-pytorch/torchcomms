!===--- test_simd_if.F90 ---------------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This is a test of the simd construct with an 'if' clause specified.
! When the statement in the 'if' clause evaluates to true,
! the preferred number of iterations secified in the simdlen clause
! will run concurrently. When the 'if' clause evaluates to be false,
! the number of iterations that will execute concurrently is one.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_simd_if

  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_simd() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_simd()
  INTEGER,DIMENSION(N):: b, c
  INTEGER:: x, errors, k
  errors = 0
  k = N

  DO x = 1, N
    b(x) = x + 5
    c(x) = 0
  END DO

  !$omp simd simdlen(64) if(k .eq. N)
  DO x = 1, N
     c(x) = b(x)
  END DO

  DO x = 1, N
     OMPVV_TEST_AND_SET_VERBOSE(errors, c(x) .ne. b(x))
  END DO

  !$omp simd simdlen(64) if(k .ne. N)
  DO x = 1, N
     c(x) = 2*b(x)
  END DO

  DO x = 1, N
     OMPVV_TEST_AND_SET_VERBOSE(errors, c(x) .ne. 2*b(x))
  END DO

  OMPVV_WARNING("Cannot guarantee vectorization by simd construct")

  test_simd = errors
  END FUNCTION test_simd
END PROGRAM test_simd_if
