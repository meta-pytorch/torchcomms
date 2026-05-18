!===--- test_target_teams_distribute_depend_in_out.F90-----------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks in-out and in-inout dependency by checking order-dependent
! results from pairs of possibly asynchronous loops. The test fails if either 
! in-out or in-inout dependency is broken.
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

  OMPVV_TEST_VERBOSE(depend_in_out() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION depend_in_out()
    INTEGER :: errors_a, errors_b, x
    INTEGER, DIMENSION(N) :: a, b, c, d

    errors_a = 0
    errors_b = 0

    DO x = 1, N
       a(x) = x
       b(x) = 2 * x
       c(x) = 0
       d(x) = 0
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map( &
    !$omp& d(1:N))
    !$omp target teams distribute nowait depend(in: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(out: c) map(alloc: &
    !$omp& b(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = c(x) + b(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors_a, d(x) .ne. 5*x)
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map( &
    !$omp from: d(1:N))
    !$omp target teams distribute nowait depend(in: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(inout: c) map(alloc: &
    !$omp& a(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = c(x) + a(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors_b, d(x) .ne. 4*x)
    END DO

    IF ((errors_a .gt. 0) .and. (errors_b .gt. 0)) THEN
       OMPVV_ERROR("Both out/inout dependencies were not dependent")
       OMPVV_ERROR("on depend(in) task")
    ELSEIF (errors_b .gt. 0) THEN
       OMPVV_ERROR("Only inout dependencies were not dependent on")
       OMPVV_ERROR("depend(in) task")
    ELSEIF (errors_a .gt. 0) THEN
       OMPVV_ERROR("Only out dependencies were not dependent on ")
       OMPVV_ERROR("depend(in) task")
    END IF

    depend_in_out = errors_a + errors_b
  END FUNCTION depend_in_out
END PROGRAM test_target_teams_distribute_depend
