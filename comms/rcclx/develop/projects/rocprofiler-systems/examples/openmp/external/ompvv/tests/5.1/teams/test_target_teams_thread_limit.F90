!===--- test_target_teams_thread_limit.F90 --------------------------===//
!
! OpenMP API Version 5.1 Nov 2020
!
! This test uses the thread_limit clause on the teams construct. The test validates 
! that only the specified threads are created by summing a shared variable across 
! all threads in a team. If the threads are correctly limited, this should produce 
! the expected value. Additional warnings are sent if specific issues occur.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_thread_limit
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_teams_thread_limit() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_teams_thread_limit()
    INTEGER :: errors(OMPVV_NUM_TEAMS_DEVICE)
    INTEGER :: num_teams, i, sum_errors
    INTEGER :: testing_thread_limit

    sum_errors = 0
    testing_thread_limit = OMPVV_NUM_THREADS_DEVICE/OMPVV_NUM_TEAMS_DEVICE
    IF (testing_thread_limit .EQ. 1) THEN
        testing_thread_limit = 2
    END IF

    DO i=1, OMPVV_NUM_TEAMS_DEVICE
        errors(i) = 0
    END DO

    !$omp target map(tofrom: num_teams,errors) 
    !$omp teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(testing_thread_limit)
    !$omp parallel
    IF ((omp_get_team_num() .EQ. 0) .AND. (omp_get_thread_num() .EQ. 0)) THEN
        num_teams = omp_get_num_teams()
    END IF 
    IF ((omp_get_thread_num() .EQ. 0) .AND. (omp_get_team_num() .LT. OMPVV_NUM_TEAMS_DEVICE)) THEN
        IF (omp_get_num_threads() .GT. testing_thread_limit) THEN
            errors(omp_get_team_num()+1) = errors(omp_get_team_num()+1) + 1
        END IF
    END IF
    !$omp end parallel
    !$omp end teams
    !$omp end target

    DO i=1, OMPVV_NUM_TEAMS_DEVICE
        sum_errors = sum_errors + errors(i)
    END DO

    OMPVV_WARNING_IF(num_teams .NE. OMPVV_NUM_TEAMS_DEVICE, "The number of teams was unexpected, the test results are likely inconclusive")
    OMPVV_WARNING_IF(testing_thread_limit .EQ. 1, "Only one thread was allocated to each team, the test results are likely inconclusive")

    test_teams_thread_limit = sum_errors
  END FUNCTION test_teams_thread_limit
END PROGRAM test_target_teams_thread_limit
