!===--- test_target_teams_distribute_reduction_add.F90----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the add operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_add() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_add()
    INTEGER,DIMENSION(N):: a, b
    INTEGER:: x, dev_sum, host_sum, errors
    errors = 0
    host_sum = 0
    dev_sum = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
    END DO

    DO x = 1, N
       host_sum = host_sum + a(x) + b(x)
    END DO

    !$omp target teams distribute defaultmap(tofrom:scalar) &
    !$omp& reduction(+:dev_sum)
    DO x = 1, N
       dev_sum = a(x) + b(x) + dev_sum
    END DO

    OMPVV_TEST_AND_SET_VERBOSE(errors, dev_sum .ne. host_sum)
    test_add = errors
  END FUNCTION test_add
END PROGRAM test_target_teams_distribute_device
