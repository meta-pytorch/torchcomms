!===--- test_target_teams_distribute_shared.F90-----------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the shared clause on a target teams distribute directive and
! tests in a few ways that the variable is shared between the teams.  In the
! first test, the atomic directive is used to indicate that all operations on
! the variable should be done atomically.  If the value is the correct value
! at the end of the region, then all teams operated on the same variable, and
! the variable was not privatized.
!
! The second test, instead of writing to the variable, only reads from the
! variable.  This tests that the value of the shared variable has not been
! initialized improperly.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_shared
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_SHARED_ENVIRONMENT
  OMPVV_TEST_VERBOSE(test_shared() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_shared()
    CHARACTER(len=300):: errMsg
    INTEGER,DIMENSION(N):: a
    INTEGER:: share, errors, x
    errors = 0
    share = 0

    DO x = 1, N
       a(x) = x
    END DO

    !$omp target teams distribute num_teams(10) shared(share) &
    !$omp& defaultmap(tofrom:scalar) map(to: a(1:N))
    DO x = 1, N
       !$omp atomic
       share = share + a(x)
    END DO

    DO x = 1, N
       share = share - a(x)
    END DO

    WRITE(errMsg, *) "Share was", share, "but expected 0"
    OMPVV_TEST_AND_SET_VERBOSE(errors, share .ne. 0)
    OMPVV_ERROR_IF(share .ne. 0, errMsg)

    share = 5

    !$omp target data map(tofrom: a(1:N)) map(tofrom: share)
    !$omp target teams distribute num_teams(10) shared(share)
    DO x = 1, N
       a(x) = a(x) + share
    END DO
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, (a(x) - 5 .ne. x))
    END DO

    test_shared = errors
  END FUNCTION test_shared
END PROGRAM test_target_teams_distribute_shared
