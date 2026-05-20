!===--- test_target_teams_distribute_depend_out_out.F90-----------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks for dependency between all combinations of out and inout
! by checking order-dependent results from pairs of possibly asynchronous
! loops. The test fails if any required dependency is broken.
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

  OMPVV_TEST_VERBOSE(depend_out_out() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION depend_out_out()
    INTEGER:: out_out_errors, inout_out_errors, out_inout_errors
    INTEGER:: inout_inout_errors, x
    INTEGER, DIMENSION(N):: a, b, c, d

    out_out_errors = 0
    inout_out_errors = 0
    out_inout_errors = 0
    inout_inout_errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = 2 * x
       c(x) = 0
       d(x) = 0
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map(&
    !$omp& from: d(1:N))
    !$omp target teams distribute nowait depend(out: c) map(alloc: &
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
       OMPVV_TEST_AND_SET(out_out_errors, d(x) .ne. 5*x)
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map(&
    !$omp& from: d(1:N))
    !$omp target teams distribute nowait depend(out: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(inout: c) map(alloc: &
    !$omp& a(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = a(x) + c(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(out_inout_errors, d(x) .ne. 4*x)
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map(&
    !$omp& from: d(1:N))
    !$omp target teams distribute nowait depend(inout: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(out: c) map(alloc: &
    !$omp& b(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = b(x) + c(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(inout_out_errors, d(x) .ne. 5*x)
    END DO

    !$omp target data map(to: a(1:N), b(1:N)) map(alloc: c(1:N)) map(&
    !$omp& from: d(1:N))
    !$omp target teams distribute nowait depend(inout: c) map(alloc: &
    !$omp& a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x) + b(x)
    END DO
    !$omp target teams distribute nowait depend(inout: c) map(alloc: &
    !$omp& a(1:N), c(1:N), d(1:N))
    DO x = 1, N
       d(x) = a(x) + c(x)
    END DO
    !$omp taskwait
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(inout_inout_errors, d(x) .ne. 4*x)
    END DO

    OMPVV_ERROR_IF(inout_inout_errors .gt. 0, "Inout task on inout task dependence failed")
    OMPVV_ERROR_IF(inout_out_errors .gt. 0, "Inout task on out task dependence failed")
    OMPVV_ERROR_IF(out_inout_errors .gt. 0, "Out task on inout task dependence failed")
    OMPVV_ERROR_IF(out_out_errors .gt. 0, "Out task on out task dependence failed")

    depend_out_out = inout_inout_errors + inout_out_errors + &
         & out_inout_errors + out_out_errors
  END FUNCTION depend_out_out
END PROGRAM test_target_teams_distribute_depend
