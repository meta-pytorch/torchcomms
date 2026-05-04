!--- test_target_teams_distribute_parallel_for_map_default.F90 --------------===//
! 
! OpenMP API Version 4.5 Nov 2015
!
! Testing the mapping of arrays and scalars, by default, the expected behavior
! is that all the arrays are mapped tofrom.
! 
!//===-----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_map_default
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_map_default() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
    INTEGER FUNCTION target_teams_distribute_parallel_for_map_default()
       INTEGER :: errors, scalar, scalar2, i , j
       INTEGER, DIMENSION(N) :: a, b, c, d
       OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_devices")
       
       errors = 0
       scalar = 20
       scalar2 = -1 
       DO i = 1, N 
          a(i) = 1
          b(i) = i 
          c(i) = 2*i
          d(i) = 0
       END DO
 
       !$omp target teams distribute parallel do
          DO j = 1, N
             d(j) = d(j) + (c(j) * (a(j) + b(j) + scalar))
          !$omp atomic write
             scalar2 = j
          END DO
 
        DO i = 1, N
           OMPVV_TEST_AND_SET(errors, d(i) .ne. ((1 + i + 20) * (2 * i)))
        END DO

   target_teams_distribute_parallel_for_map_default = errors
   END FUNCTION target_teams_distribute_parallel_for_map_default
END PROGRAM test_target_teams_distribute_parallel_for_map_default  
