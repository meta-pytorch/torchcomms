!===--- test_target_teams_distribute_default_private.F90--------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the default private clause on a target teams distribute
! directive to indicate that a scalar variable should be by default made
! private to each team executing the teams distribute region.  The test then
! operates on the privatized variable in such a way that would most likely
! cause competiting operations if the variable is not privatized.  If the
! computation completes without errors, we assume that the privatization
! occured.
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
  OMPVV_TEST_VERBOSE(test_default_private() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_default_private()
    INTEGER,DIMENSION(N):: a, b, c, d
    INTEGER:: privatized, errors, x, y
    errors = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 2*x
       d(x) = 0
    END DO

    !$omp target teams distribute default(private) shared(a, &
    !$omp& b, c, d) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       privatized = 0
       DO y = 1, a(x) + b(x)
          privatized = privatized + 1
       END DO
       d(x) = c(x)*privatized
    END DO

    DO x = 1, N
       OMPVV_TEST_AND_SET(errors, (d(x) .ne. (2*x)*(1 + x)))
    END DO

    test_default_private = errors
  END FUNCTION test_default_private
END PROGRAM test_target_teams_distribute_device
