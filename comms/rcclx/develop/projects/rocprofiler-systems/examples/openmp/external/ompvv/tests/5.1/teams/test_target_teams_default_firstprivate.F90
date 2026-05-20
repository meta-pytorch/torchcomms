!===--- test_target_teams_default_firstprivate.F90 --------------------------===//
!
! OpenMP API Version 5.1 Nov 2020
!
! This test uses the default(firstprivate) clause on a teams directive. The test
! validates that when the default(firstprivate) clause is present all
! variables without explicit sharing rules are not avaialble outside the region
! and are private to each team. The default(firstprivate) clause is tested using
! the not_shared variable, whose value should have not changed after the target
! teams construct since all changes to the firstprivate variable should not persist
! after the construct. Additionally if there is a race condition, we know the 
! variable is not defaulting to firstprivate either.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_firstprivate
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_teams_firstprivate() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_teams_firstprivate()
    INTEGER :: errors, i
    INTEGER :: not_shared, num_teams

    errors = 0
    not_shared = 5
    num_teams = 0

    !$omp target teams default(firstprivate) map(tofrom: num_teams,errors) shared(num_teams,errors) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    IF (omp_get_team_num() .EQ. 0) then
        num_teams = omp_get_num_teams()
        ! check first private
        IF (not_shared .NE. 5) then
          errors = errors + 1
        END IF
    END IF 
    DO i=1, omp_get_num_teams()
        not_shared = not_shared + 5
    END DO
    !$omp end target teams

    OMPVV_WARNING_IF(num_teams .NE. OMPVV_NUM_TEAMS_DEVICE, "Number of teams was unexpected, test results likely inconclusive")
    OMPVV_TEST_AND_SET(errors, (not_shared .NE. 5))

    test_teams_firstprivate = errors
  END FUNCTION test_teams_firstprivate
END PROGRAM test_target_teams_firstprivate
