!===--- test_target_teams_distribute_default_shared.F90---------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the default(shared) clause on a target teams distribute
! directive.  The test aims to validate that when the default(shared) clause
! is present, all variables without explicit data sharing attributes will
! be shared within the region.  To test this, we test that a data element
! that should be shared due to the default(shared) clause is available to
! all the teams.  The first test uses atomic to write to the variable without
! race conditions.  The second test uses synchronization constructs to have
! one thread change the shared variable and ensures all threads see the change.
!
!===------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_default_shared
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  OMPVV_TEST_OFFLOADING
  errors = 0

  OMPVV_TEST_VERBOSE(default_shared1() .ne. 0)
  OMPVV_TEST_VERBOSE(default_shared2() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION default_shared1()
    INTEGER :: a(N)
    INTEGER :: share, errors, num_teams, x
    errors = 0
    share = 0

    DO x = 1, N
       a(x) = x
    END DO

    !$omp target data map(to: a(1:N)) map(tofrom: share, num_teams)
    !$omp target teams distribute default(shared) defaultmap(tofrom:scalar) &
    !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       IF (omp_get_team_num() .eq. 0) THEN
          num_teams = omp_get_num_teams()
       END IF
       !$omp atomic
       share = share + a(x)
    END DO
    !$omp end target data

    DO x = 1, N
       share = share - x
    END DO

    OMPVV_WARNING_IF(num_teams .eq. 1, "Test operated on one team, results of default shared test are inconclusive.")

    OMPVV_TEST_AND_SET(errors, share .ne. 0)
    default_shared1 = errors
  END FUNCTION default_shared1

  INTEGER FUNCTION default_shared2()
    INTEGER :: a(N)
    INTEGER :: share, errors, x
    errors = 0
    share = 5

    DO x = 1, N
       a(x) = x
    END DO

    !$omp target data map(tofrom: a(1:N), share)
    !$omp target teams distribute default(shared) defaultmap(tofrom:scalar) &
    !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       a(x) = a(x) + share
    END DO
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET(errors, a(x) .ne. x + 5)
    END DO
    default_shared2 = errors
  END FUNCTION default_shared2
END PROGRAM test_target_teams_distribute_default_shared
