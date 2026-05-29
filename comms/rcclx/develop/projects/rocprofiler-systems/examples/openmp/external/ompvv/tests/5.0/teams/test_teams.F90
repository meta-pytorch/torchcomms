!===--- test_teams.F90 -----------------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the teams directive on host and verifies the 
! requested number of teams with requested number of threads were
! created.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 4
#define K 2

PROGRAM test_teams
 
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(teams() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
    INTEGER FUNCTION teams()
    INTEGER,DIMENSION(OMPVV_NUM_TEAMS_DEVICE):: num_teams 
    INTEGER,DIMENSION(OMPVV_NUM_THREADS_DEVICE):: num_threads
    INTEGER,DIMENSION(K):: errors
    INTEGER:: x
    errors(1) = 0
    errors(2) = 0


    DO x = 1, OMPVV_NUM_TEAMS_DEVICE
      num_teams(x) = -99
    END DO

    DO x = 1, OMPVV_NUM_THREADS_DEVICE
      num_threads(x) = -99
    END DO

    !$omp teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(OMPVV_NUM_THREADS_DEVICE)
      num_teams(omp_get_team_num() + 1) = omp_get_num_teams()
      !$omp parallel master
        num_threads(omp_get_team_num() + 1) = omp_get_num_threads()
      !$omp end parallel master
    !$omp end teams 
  
    OMPVV_WARNING_IF(num_teams(1) .eq. 1, "Test operated with one team, num_teams requested is inconsistent with this result")

    OMPVV_ERROR_IF(num_teams(1) .le. 1, "omp_get_num_teams() reported a value less than one.")

    OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Team 0 reported only 1 thread. This is inconsistent with the thread limit that has been set.")

    OMPVV_ERROR_IF(num_threads(1) .le. 1, "omp_get_num_threads() reported a value below one.")

    DO x = 2, num_teams(1)
      IF (num_teams(x) .ne. num_teams(x-1)) THEN
        errors(1) = errors(1) + 1
      END IF
      IF (num_threads(x) .ne. num_threads(x-1)) THEN
        errors(2) = errors(2) + 1
      END IF
    END DO

    teams = errors(1) + errors(2)
    END FUNCTION teams
END PROGRAM test_teams 


