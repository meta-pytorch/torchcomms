!===--- test_target_teams_distribute_depend_list.F90-----------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks for dependency between multiple out-dependent tasks by
! checking order-dependent results from pairs of possibly asynchronous loops
! The test fails if any required dependency is broken.
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

  OMPVV_TEST_VERBOSE(depend_list() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION depend_list()
    INTEGER:: errors, x
    INTEGER,DIMENSION(N):: a, b, c, d, e, f, g

    errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = 2 * x
       c(x) = 0
       d(x) = 0
       e(x) = 0
       f(x) = 0
       g(x) = 0
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N), d(1:N), &
    !$omp& e(1:N)) map(from: f(1:N), g(1:N))
    !$omp target teams distribute nowait depend(out: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(out: d) map(alloc: &
    !$omp a(1:N), b(1:N), d(1:N))
    DO x = 1, N
       d(x) = a(x) + b(x) + x
    END DO
    !$omp target teams distribute nowait depend(out: c, d, e) map(alloc:&
    !$omp& c(1:N), d(1:N), e(1:N))
    DO x = 1, N
       e(x) = c(x) + d(x)
    END DO
    !$omp target teams distribute nowait depend(out: e) map(alloc: &
    !$omp& a(1:N), e(1:N), f(1:N))
    DO x = 1, N
       f(x) = e(x) + a(x)
    END DO
    !$omp target teams distribute nowait depend(out: e) map(alloc: &
    !$omp& b(1:N), e(1:N), g(1:N))
    DO x = 1, N
       g(x) = e(x) + b(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, (f(x) .ne. 8 * x) .or. (g(x) .ne. 9 * x))
    END DO

    depend_list = errors
  END FUNCTION depend_list
END PROGRAM test_target_teams_distribute_depend
