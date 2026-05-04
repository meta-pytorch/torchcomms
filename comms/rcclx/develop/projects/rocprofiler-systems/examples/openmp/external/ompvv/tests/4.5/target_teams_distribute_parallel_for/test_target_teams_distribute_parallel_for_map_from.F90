!--- test_target_teams_distribute_parallel_for_map_from.F90 -----------------===//
! 
! OpenMP API Version 4.5 Nov 2015
!
! Testing the mapping of arrays and scalars through the map clause with
! the 'from' map-type.
! 
!//===-----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1000

PROGRAM test_target_teams_distribute_parallel_for_map_from
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_map_from() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION target_teams_distribute_parallel_for_map_from()
     INTEGER :: errors, scalar, i, j
     INTEGER, DIMENSION(N) :: a
     
     errors = 0     
     scalar = 0
     
     DO i = 1, N
        a(i) = 1
     END DO

     !$omp target teams distribute parallel do map(from: a, scalar)
        DO j = 1, N
        !$omp atomic write
           scalar = 20
           a(j) = 10
        END DO

     OMPVV_TEST_AND_SET(errors, scalar .ne. 20)
     DO i = 1, N
        OMPVV_TEST_AND_SET(errors, a(i) .ne. 10)
     END DO

     target_teams_distribute_parallel_for_map_from = errors
  END FUNCTION target_teams_distribute_parallel_for_map_from
END PROGRAM test_target_teams_distribute_parallel_for_map_from
