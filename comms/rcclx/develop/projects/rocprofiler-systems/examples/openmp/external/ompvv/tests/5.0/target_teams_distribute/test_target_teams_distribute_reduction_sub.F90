!===--- test_target_teams_distribute_reduction_sub.F90----------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the subtract operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_sub() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_sub()
    INTEGER,DIMENSION(N):: a
    INTEGER:: x, y, errors, host_result, result
    errors = 0

    DO y = 1, N
       result = 0
       host_result = 0

       DO x = 1, N
          a(x) = x + y
       END DO

       DO x = 1, N
          host_result = host_result - a(x)
       END DO

       !$omp target teams distribute map(to: a(1:N)) reduction(-:result) &
       !$omp& map(tofrom: result)
       DO x = 1, N
          result = result - a(x)
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .ne. host_result)
    END DO

    test_sub = errors
  END FUNCTION test_sub
END PROGRAM test_target_teams_distribute_device
