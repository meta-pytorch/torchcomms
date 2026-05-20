!===--- test_target_teams_distribute_reduction_max.F90----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the max operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_max() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_max()
    INTEGER,DIMENSION(N):: a
    INTEGER:: result, x, y, errors
    errors = 0

    DO y = 1, 32
       DO x = 1, N
          a(x) = x + y
       END DO
       result = a(1)

       !$omp target teams distribute defaultmap(tofrom:scalar) &
       !$omp& reduction(max:result)
       DO x = 1, N
          result = max(a(x), result)
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .ne. y + N)
    END DO

    test_max = errors
  END FUNCTION test_max
END PROGRAM test_target_teams_distribute_device
