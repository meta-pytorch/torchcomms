!===--- test_target_teams_distribute_reduction_bitxor.F90-------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the bitxor operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_bitxor() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_bitxor()
    REAL(8),DIMENSION(N):: r
    INTEGER,DIMENSION(N):: a
    INTEGER:: result, host_result, x, y, errors
    errors = 0

    CALL RANDOM_SEED()

    DO y = 1, 32
       CALL RANDOM_NUMBER(r)
       host_result = 0
       result = 0

       DO x = 1, N
          a(x) = INT(r(x) * 2)
          host_result = ieor(a(x), host_result)
       END DO

       !$omp target teams distribute map(to: a(1:N)) &
       !$omp& reduction(ieor:result) map(tofrom: result)
       DO x = 1, N
          result = ieor(a(x), result)
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .ne. host_result)
    END DO

    test_bitxor = errors
  END FUNCTION test_bitxor
END PROGRAM test_target_teams_distribute_device
