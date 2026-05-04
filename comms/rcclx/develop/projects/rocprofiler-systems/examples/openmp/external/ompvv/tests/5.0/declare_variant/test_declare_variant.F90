!===--- test_declare_variant.F90 --------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the declare variant clause to create two variants of a
! simple base function, one for use in a parallel region, and one for use
! in a target region. The function sets each element of an array to its
! index. Each variant is called on a separate array and the results are
! checked.
!
! This test is derived from OpenMP 5.0 Examples Doc - declare_variant.1.f90
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

MODULE variants
  use omp_lib
CONTAINS
  SUBROUTINE fn(arr)
    INTEGER,INTENT(inout) :: arr(N)
    INTEGER :: i
    !$omp declare variant(p_fn) match(construct = {parallel})
    !$omp declare variant(t_fn) match(construct = {target})

    DO i = 1, N
       arr(i) = i
    END DO
  END SUBROUTINE fn

  SUBROUTINE p_fn(arr)
    INTEGER,INTENT(inout) :: arr(N)
    INTEGER :: i

    !$omp do
    DO i = 1, N
       arr(i) = i + 1
    END DO
  END SUBROUTINE p_fn

  SUBROUTINE t_fn(arr)
    INTEGER,INTENT(inout) :: arr(N)
    INTEGER :: i
    !$omp declare target

    !$omp distribute simd
    DO i = 1, N
       arr(i) = i + 2
    END DO
  END SUBROUTINE t_fn
END MODULE variants

PROGRAM test_declare_variant
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  USE variants
  implicit none
  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_body() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_body()
    INTEGER :: default_errors, p_errors, t_errors, x
    INTEGER,DIMENSION(N) :: a, b, c

    DO x = 1, N
       a(x) = 0
       b(x) = 0
       c(x) = 0
    END DO

    default_errors = 0
    p_errors = 0
    t_errors = 0

    call fn(a)

    !$omp parallel
    call fn(b)
    !$omp end parallel

    !$omp target teams map(tofrom: c)
    call fn(c)
    !$omp end target teams

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(default_errors, a(x) .ne. x)
       OMPVV_TEST_AND_SET_VERBOSE(p_errors, b(x) .ne. (x + 1))
       OMPVV_TEST_AND_SET_VERBOSE(t_errors, c(x) .ne. (x + 2))
    END DO

    OMPVV_ERROR_IF(default_errors .ne. 0, "Did not use default variant of test function when expected.")
    OMPVV_ERROR_IF(p_errors .ne. 0, "Did not use parallel variant of test function when expected.")
    OMPVV_ERROR_IF(t_errors .ne. 0, "Did not use target variant of test function when expected.")

    test_body = default_errors + p_errors + t_errors
  END FUNCTION test_body
END PROGRAM test_declare_variant
