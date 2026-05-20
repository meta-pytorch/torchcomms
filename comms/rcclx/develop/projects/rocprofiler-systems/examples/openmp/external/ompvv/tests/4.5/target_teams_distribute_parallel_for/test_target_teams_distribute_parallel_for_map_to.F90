!--- test_target_teams_distribute_parallel_for_map_to.F90 -------------------===//
! 
! OpenMP API Version 4.5 Nov 2015
!
! Testing the mapping of arrays and scalars through the map clause with
! the 'to' map-type.
! 
!//===-----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1000

PROGRAM test_target_teams_distribute_parallel_for_map_to
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_map_to() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
    INTEGER FUNCTION target_teams_distribute_parallel_for_map_to()
       INTEGER :: errors, scalar, i , j
       INTEGER, DIMENSION(N) :: a, b, d

       scalar = 50
       errors = 0
       DO i = 1, N
          a(i) = 1
          b(i) = i
          d(i) = 0
       END DO

       !$omp target teams distribute parallel do map(to: a, b, scalar)&
       !$omp& map(tofrom: d)
          DO j = 1, N
             d(j) = (a(j) + b(j)) * scalar
          END DO

       DO i = 1, N
          OMPVV_TEST_AND_SET(errors, d(i) .ne. ((1+i) * 50))
       END DO
          
       target_teams_distribute_parallel_for_map_to = errors
    END FUNCTION target_teams_distribute_parallel_for_map_to
END PROGRAM test_target_teams_distribute_parallel_for_map_to
