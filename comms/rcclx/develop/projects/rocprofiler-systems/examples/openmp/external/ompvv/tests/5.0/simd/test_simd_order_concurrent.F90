!===--- test_simd_order_concurrent.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks for support of the order(concurrent) clause on a simd construct.
! When an order(concurrent) clause is present on a simd construct, all of the same 
! restrictions from having a loop construct with an order(concurrent) also apply.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_simd_order_concurrent

  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_simd_order() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_simd_order()
  INTEGER,DIMENSION(N):: b, c
  INTEGER:: i, errors
  TYPE new_struct
  INTEGER,DIMENSION(N):: a
  END TYPE new_struct
  TYPE(new_struct):: struct_t

  errors = 0

  DO i = 1, N
    struct_t%a(i) = i
    b(i) = i + 5
    c(i) = 0
  END DO

  !$omp simd order(concurrent)
  DO i = 1, N
     c(i) = struct_t%a(i) * b(i)
  END DO

  DO i = 1, N
     OMPVV_TEST_AND_SET(errors, c(i) .NE. struct_t%a(i) * b(i))
  END DO

  test_simd_order = errors
  END FUNCTION test_simd_order
END PROGRAM test_simd_order_concurrent
