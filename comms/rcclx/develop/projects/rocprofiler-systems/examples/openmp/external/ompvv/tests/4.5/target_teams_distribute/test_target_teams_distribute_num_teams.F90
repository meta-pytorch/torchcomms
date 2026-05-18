!===--- test_target_teams_distribute_num_teams.F90--------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the num_teams clause on a target teams distribute directive to
! indicate a requested number of teams to execute the teams distribute region.
! The specifications indicate that the number of teams that are given can be any
! number that is equal to or less than the indicated value.  We first run a
! target teams distribute region without the clause to see what the default
! number of teams is, and then we use a value that is less than that in the
! test of the num_teams clause.  If the region is run with more teams than
! indicated, the test errors.  If the region is run with less teams than
! indicated, the test issues a warning since it is known that the device can
! run with more teams than was actually given.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_num_teams
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  errors = 0

  OMPVV_TEST_VERBOSE(num_teams() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION num_teams()
    INTEGER:: errors, default_num_teams, x
    INTEGER,DIMENSION(N):: a, b, c, num_team

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 0
       num_team(x) = -1
    END DO

    errors = 0
    default_num_teams = -1

    !$omp target teams distribute map(tofrom: default_num_teams, c(1:N)) &
    !$omp& map(to: a(1:N), b(1:N))
    DO x = 1, N
       IF (omp_get_team_num() .eq. 0) THEN
          default_num_teams = omp_get_num_teams()
       END IF
       c(x) = a(x) + b(x)
    END DO

    IF (default_num_teams .eq. 1) THEN
       OMPVV_WARNING("Test operated with one team. Testing num_teams")
       OMPVV_WARNING("cannot be done.")
    ELSEIF (default_num_teams .le. 0) THEN
       OMPVV_ERROR("omp_get_num_teams() returned result less than or")
       OMPVV_ERROR("equal to 0.")
    ELSE
       !$omp target teams distribute num_teams(default_num_teams / 2) &
       !$omp& map(to: a(1:N), b(1:N)) map(from: c(1:N), num_team(1:N))
       DO x = 1, N
          c(x) = a(x) + b(x)
          num_team(x) = omp_get_num_teams()
       END DO

       DO x = 2, N
          IF (num_team(x) .ne. num_team(x - 1)) THEN
             errors = errors + 1
             OMPVV_ERROR("omp_get_num_teams returned an inconsistent number of")
             OMPVV_ERROR("teams between iterations.")
             exit
          END IF
       END DO

       IF (num_team(1) .gt. default_num_teams / 2) THEN
          errors = errors + 1
          OMPVV_ERROR("Test ran on more teams than requested")
       ELSEIF (num_team(1) .lt. default_num_teams / 2) THEN
          OMPVV_WARNING("Test ran on less teams than requested. This ")
          OMPVV_WARNING("is still spec-conformant.")
       END IF
    END IF
    num_teams = errors
  END FUNCTION num_teams
END PROGRAM test_target_teams_distribute_num_teams
