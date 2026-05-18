!--- test_target_teams_distribute_parallel_for_map_tofrom.F90 ---------------===//
! 
! OpenMP API Version 4.5 Nov 2015
!
! Testing the mapping of arrays and scalars through the map clause with
! the 'tofrom' map-type.
! 
!//===-----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1000

PROGRAM test_target_teams_distribute_parallel_for_map_tofrom
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_map_tofrom() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_teams_distribute_parallel_for_map_tofrom()
      INTEGER :: scalar_to, scalar_from, errors, i, j 
      INTEGER, DIMENSION(N) :: a, b, c, d
      OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_map_tofrom");
      
      scalar_to = 50
      scalar_from = 50
      errors = 0
      
      DO i = 1, N
         a(i) = 1
         b(i) = i
         c(i) = 2*i
         d(i) = 0
      END DO

      !$omp target teams distribute parallel do map(tofrom: a, b, c, d,&
      !$omp& scalar_to, scalar_from)
         DO j = 1, N
            d(j) = d(j) + c(j) * (a(j) + b(j) + scalar_to)
            a(j) = 10
            b(j) = 11
            c(j) = 12
         !$omp atomic write
            scalar_from = 13
         END DO

      OMPVV_TEST_AND_SET(errors, scalar_from .ne. 13)
      DO i = 1, N
         OMPVV_TEST_AND_SET(errors, a(i) .ne. 10)
         OMPVV_TEST_AND_SET(errors, b(i) .ne. 11)
         OMPVV_TEST_AND_SET(errors, c(i) .ne. 12)
         OMPVV_TEST_AND_SET(errors, d(i) .ne. ((1 + i + 50) * 2 * i))
      END DO

   target_teams_distribute_parallel_for_map_tofrom = errors
   END FUNCTION target_teams_distribute_parallel_for_map_tofrom
END PROGRAM test_target_teams_distribute_parallel_for_map_tofrom

