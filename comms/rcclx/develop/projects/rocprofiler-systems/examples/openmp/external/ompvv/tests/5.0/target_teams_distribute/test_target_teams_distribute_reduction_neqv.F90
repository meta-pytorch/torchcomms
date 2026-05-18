!===--- test_target_teams_distribute_reduction_neqv.F90---------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the neqv operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_neqv() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_neqv()
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
          host_result = a(x) .neqv. host_result
       END DO

       !$omp target teams distribute map(to: a(1:N)) reduction(.neqv.: &
       !$omp& result) map(tofrom: result)
       DO x = 1, N
          result = a(x) .neqv. result
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .neqv. host_result)
    END DO

    test_neqv = errors
  END FUNCTION test_neqv
END PROGRAM test_target_teams_distribute_device
