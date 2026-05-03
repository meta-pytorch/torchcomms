!===--- test_target_teams_distribute_reduction_multiply.F90-----------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the multiply operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_multiply() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_multiply()
    REAL(8),DIMENSION(N):: r
    INTEGER,DIMENSION(N):: a
    INTEGER:: x, y, errors, device_result, host_result

    CALL RANDOM_SEED()
    CALL RANDOM_NUMBER(r)

    errors = 0

    DO x = 1, N
       a(x) = INT(1 + (r(x) * 2))
    END DO

    DO x = 1, N, 16
       device_result = 1
       !$omp target teams distribute defaultmap(tofrom:scalar) &
       !$omp& reduction(*:device_result)
       DO y = 1, 15
          device_result = a(x + y) * device_result
       END DO
       host_result = 1
       DO y = 1, 15
          host_result = a(x + y) * host_result
       END DO
       OMPVV_TEST_AND_SET_VERBOSE(errors, host_result .ne. device_result)
    END DO

    test_multiply = errors
  END FUNCTION test_multiply
END PROGRAM test_target_teams_distribute_device
