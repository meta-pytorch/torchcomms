!===--- test_simd_nontemporal.F90 ------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks for support of the nontemporal clause on a simd construct. 
! The nontemporal clause indicates that accesses to the storage location of list 
! items have low temporal locality across the iterations in which those storage 
! locations are accessed. 
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1028
#define STRIDE_LEN 100

PROGRAM test_simd_nontemporal

  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_simd_nontemp() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_simd_nontemp()
  INTEGER,DIMENSION(N):: a, b, c
  INTEGER:: i, errors

  errors = 0

  DO i = 1, N
    a(i) = 10
    b(i) = i
    c(i) = 2 * i
  END DO

  !$omp simd nontemporal(a, b, c)
  DO i = 1, N, STRIDE_LEN
     a(i) = b(i) * c(i)
  END DO

  DO i = 1, N
     IF ( MOD(i, STRIDE_LEN) .EQ. 1 ) THEN
       OMPVV_TEST_AND_SET(errors, a(i) .NE. (b(i) * c(i)))
     ELSE
       OMPVV_TEST_AND_SET(errors, a(i) .NE. 10)
     END IF
  END DO

  test_simd_nontemp = errors
  END FUNCTION test_simd_nontemp
END PROGRAM test_simd_nontemporal
