!===--- test_target_teams_distribute_parallel_for_collapse.F90----------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Testing loop collapse with multiple loops. Trying to mimic lsms kernel. 
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define SIZE_N 10
#define SIZE_M 12

PROGRAM test_target_teams_distribute_parallel_for_collapse
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  errors = 0

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_collapse() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION target_teams_distribute_parallel_for_collapse()
    INTEGER,DIMENSION(SIZE_N, SIZE_M, SIZE_N, SIZE_M):: a
    INTEGER:: i, j, k, l, errors, temp
    errors = 0

    ! a array initialization
    DO i = 1, SIZE_N
    DO j = 1, SIZE_M
    DO k = 1, SIZE_N
    DO l = 1, SIZE_M
       a(i,j,k,l) = 1
    END DO
    END DO
    END DO
    END DO

    !$omp target teams distribute parallel do collapse(4) map(tofrom: a) &
    !$omp& private(i,j,k,l)
    DO i = 1, SIZE_N
    DO j = 1, SIZE_M
    DO k = 1, SIZE_N
    DO l = 1, SIZE_M
       a(i,j,k,l) = a(i,j,k,l) + i+2*j+3*k+4*l
    END DO
    END DO
    END DO
    END DO

    DO i = 1, SIZE_N
    DO j = 1, SIZE_M
    DO k = 1, SIZE_N
    DO l = 1, SIZE_M
       temp = 1 + i + 2*j + 3*k + 4*l
       OMPVV_TEST_AND_SET(errors, a(i,j,k,l) .ne. temp)
    END DO
    END DO
    END DO
    END DO

    target_teams_distribute_parallel_for_collapse = errors
  END FUNCTION target_teams_distribute_parallel_for_collapse
END PROGRAM test_target_teams_distribute_parallel_for_collapse
