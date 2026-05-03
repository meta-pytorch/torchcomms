!===--- test_target_teams_distribute_depend_unused_data.F90-----------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks if out-out dependency works even if the data in the list
! is unused by either task. Both tasks are given the nowait clause to allow
! for the possibility that they will be incorrectly run out of order. If the
! two target teams ditribute loops run out of order, the test fails.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_depend
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  OMPVV_TEST_OFFLOADING
  errors = 0

  OMPVV_TEST_VERBOSE(depend_unused_data() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION depend_unused_data()
    INTEGER:: errors, x
    INTEGER,DIMENSION(N):: a, b, c, d
    INTEGER,DIMENSION(1):: random_data

    errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = 2 * x
       c(x) = 0
       d(x) = 0
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N), &
    !$omp& random_data(1:1)) map(from: d(1:N))
    !$omp target teams distribute nowait depend(out: random_data) &
    !$omp map(alloc: a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(out: random_data) &
    !$omp& map(alloc: d(1:N), c(1:N), b(1:N))
    DO x = 1, N
       d(x) = c(x) + b(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, d(x) .ne. 5*x)
    END DO

    depend_unused_data = errors
  END FUNCTION depend_unused_data
END PROGRAM test_target_teams_distribute_depend
