!===--- test_target_teams_distribute.F90 -----------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the target teams distribute directive and tests to validate
! that computation inside the region executes properly.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_SHARED_ENVIRONMENT

  OMPVV_TEST_VERBOSE(test_distribute() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_distribute()
    CHARACTER(len=300):: infoMsg
    INTEGER,DIMENSION(N):: num_teams, a, b
    INTEGER:: errors, x
    errors = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
       num_teams(x) = -1
    END DO

    !$omp target teams distribute map(tofrom: a(1:N), num_teams(1:N)) map(to: b(1:N))
    DO x = 1, N
       num_teams(x) = omp_get_num_teams();
       a(x) = a(x) + b(x)
    END DO

    OMPVV_WARNING_IF(num_teams(1) .eq. 1, "Test ran with one team, can't guarantee parallelism of teams")

    OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams(1) .lt. 1)

    IF (errors .eq. 0) THEN
       DO x = 2, N
          OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams(x) .ne. num_teams(x - 1))
          OMPVV_ERROR_IF(num_teams(x) .ne. num_teams(x - 1), "Test reported an inconsistent number of teams")
          OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) .ne. 1 + b(x))
          IF (errors .gt. 0) THEN
             exit
          END IF
       END DO
    END IF

    WRITE(infoMsg, *) "Test passed with", num_teams(1), "teams."
    OMPVV_INFOMSG_IF(errors .eq. 0, infoMsg)

    test_distribute = errors
  END FUNCTION test_distribute
END PROGRAM test_target_teams_distribute
