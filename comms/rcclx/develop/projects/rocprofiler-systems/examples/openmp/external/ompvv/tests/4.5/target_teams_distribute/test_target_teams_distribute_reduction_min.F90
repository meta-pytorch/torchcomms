!===--- test_target_teams_distribute_reduction_min.F90----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the min operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_min() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_min()
    INTEGER,DIMENSION(N):: a
    INTEGER:: result, x, y, errors
    errors = 0

    DO y = 1, 32
       DO x = 1, N
          a(x) = x + y
       END DO

       result = a(N)

       !$omp target teams distribute defaultmap(tofrom:scalar) &
       !$omp& reduction(min:result)
       DO x = 1, N
          result = min(a(x), result)
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .ne. 1 + y)
    END DO

    test_min = errors
  END FUNCTION test_min
END PROGRAM test_target_teams_distribute_device
