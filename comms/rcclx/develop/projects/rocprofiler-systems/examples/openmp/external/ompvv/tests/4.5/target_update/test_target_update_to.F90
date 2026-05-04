!===--- test_target_update_to.F90 -------------------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks target update with the 'to' clause by checking that
! after a target update to is applied to an array, the device reads and
! uses the expected, new value rather than the previous value.
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_update_to
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_OFFLOADING
   OMPVV_TEST_VERBOSE(target_update_to() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_update_to()
      INTEGER :: errors, i
      INTEGER, DIMENSION(N) :: a, b, c
      i = 0
      errors = 0

      DO i = 1, N
         a(i) = 10
         b(i) = 2
         c(i) = 0
      END DO

      !$omp target data map(to: a, b) map(from: c)
         !$omp target
            DO i = 1, N
               c(i) = a(i) + b(i)
            END DO
         !$omp end target
      !$omp end target data

      !update b
      DO i = 1, N
         b(i) = b(i) * 2
      END DO
      !$omp target update to(b)

      !$omp target
         DO i = 1, N
            c(i) = c(i) + b(i)
         END DO
      !$omp end target

      DO i = 1, N
         OMPVV_TEST_AND_SET_VERBOSE(errors, c(i) .ne. 16)
      END DO  

      target_update_to = errors       
   END FUNCTION target_update_to
END PROGRAM test_target_update_to
