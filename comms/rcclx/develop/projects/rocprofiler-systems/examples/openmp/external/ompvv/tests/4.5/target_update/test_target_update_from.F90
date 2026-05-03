!===--- test_target_update_from.F90 -----------------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks the target update motion clause 'from' by mapping an array 
! to the device with map-type 'to', changing the values of array on the device,
! and finally using the update 'from' motion clause to assign the value of the 
! list item. Back on the host, measures are taken to ensure the value was properly
! updated.   
! 
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_update_from
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_OFFLOADING

   OMPVV_TEST_VERBOSE(target_update_from() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_update_from()
      INTEGER :: errors, i
      INTEGER, DIMENSION(N) :: a, b, c 
      errors = 0
      i = 0

      DO i = 1, N
         a(i) = 10
         b(i) = 2
         c(i) = 0
      END DO

      !$omp target enter data map(to: a, b)
      !$omp target
         DO i = 1, N
            b(i) = (a(i) + b(i))
         END DO
      !$omp end target

      !$omp target update from(b)
     
      DO i = 1, N
         OMPVV_TEST_AND_SET_VERBOSE(errors, b(i) .ne. 12)
      END DO

      !$omp target
         DO i = 1, N
            c(i) = (2 * b(i))
         END DO
      !$omp end target

      DO i = 1, N
         OMPVV_TEST_AND_SET_VERBOSE(errors, c(i) .ne. 24)
      END DO

      target_update_from = errors

   END FUNCTION target_update_from
END PROGRAM test_target_update_from
