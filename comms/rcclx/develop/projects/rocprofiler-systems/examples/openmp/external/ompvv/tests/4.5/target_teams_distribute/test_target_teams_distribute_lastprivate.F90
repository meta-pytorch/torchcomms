!===--- test_target_teams_distribute_lastprivate.F90------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the lastprivate clause to indicate that the privatized value
! that is passed as the parameter should also be returned with the value that
! results from the thread that runs the last iteration of the for loop in the
! target teams distribute directive.  The clause can be used with both scalar
! and array data types and both situations are tested.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_lastprivate
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  OMPVV_TEST_OFFLOADING
  errors = 0

  OMPVV_TEST_VERBOSE(test_scalar() .ne. 0)
  OMPVV_TEST_VERBOSE(test_array() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_scalar()
    INTEGER:: x, errors, privatized
    INTEGER,DIMENSION(N):: a, b, c

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 0
    END DO

    errors = 0

    !$omp target data map(to: a(1:N), b(1:N)) map(tofrom: c(1:N))
    !$omp target teams distribute lastprivate(privatized) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N)) defaultmap(tofrom:scalar)
    DO x = 1, N
       privatized = a(x) - b(x)
       c(x) = privatized + b(x)
    END DO
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, c(x) .ne. a(x))
    END DO

    OMPVV_TEST_AND_SET_VERBOSE(errors, privatized .ne. (a(N) - b(N)))

    test_scalar = errors
  END FUNCTION test_scalar

  INTEGER FUNCTION test_array()
    INTEGER:: x, errors
    INTEGER,DIMENSION(N):: a, b, c
    INTEGER,DIMENSION(2):: privatized

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = MOD(x, 10)
    END DO

    errors = 0

    !$omp target data map(to: a(1:N), b(1:N), c(1:N)) map(tofrom: &
    !$omp& privatized(1:2))
    !$omp target teams distribute lastprivate(privatized) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       privatized(1) = a(x) + b(x) + c(x)
       privatized(2) = (a(x) + b(x)) * c(x)
    END DO
    !$omp end target data

    OMPVV_TEST_AND_SET_VERBOSE(errors, privatized(1) .ne. a(N) + b(N) + c(N))
    OMPVV_TEST_AND_SET_VERBOSE(errors, privatized(2) .ne. (a(N) + b(N)) * c(N))

    test_array = errors
  END FUNCTION test_array
END PROGRAM test_target_teams_distribute_lastprivate
