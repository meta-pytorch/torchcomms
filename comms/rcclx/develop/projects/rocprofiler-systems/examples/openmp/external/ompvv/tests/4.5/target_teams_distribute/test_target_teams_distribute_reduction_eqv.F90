!===--- test_target_teams_distribute_reduction_eqv.F90----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the eqv operator, that the variable in the
! reduction clause is properly reduced.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  errors = 0

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_eqv() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_eqv()
    LOGICAL,DIMENSION(N):: a
    REAL(8),DIMENSION(N):: randoms
    INTEGER:: x, y, errors
    LOGICAL:: host_result, result
    errors = 0
    CALL RANDOM_SEED()

    DO y = 1, 32
       CALL RANDOM_NUMBER(randoms)
       a = .TRUE.
       host_result = .TRUE.
       result = .TRUE.
       DO x = 1, N
          IF (randoms(x) .gt. .5) THEN
             a(x) = .FALSE.
          END IF
       END DO

       DO x = 1, N
          host_result = a(x) .eqv. host_result
       END DO

       !$omp target teams distribute reduction(.eqv.: &
       !$omp& result) defaultmap(tofrom:scalar)
       DO x = 1, N
          result = a(x) .eqv. result
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .neqv. host_result)
    END DO

    test_eqv = errors
  END FUNCTION test_eqv
END PROGRAM test_target_teams_distribute_device
