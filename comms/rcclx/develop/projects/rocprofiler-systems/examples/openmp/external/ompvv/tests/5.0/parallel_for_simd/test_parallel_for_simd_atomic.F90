!===--- test_parallel_for_simd_atomic.F90 ----------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks that the atomic construct can be used within the parallel
! for simd construct to avoid a race condition in updating a shared
! variable, whose value is checked after updating.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_parallel_for_simd_atomic
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_simd_atomic() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_simd_atomic()
    INTEGER:: errors, x, i

    errors = 0
    x = 0

    !$omp parallel do simd shared(x) &
    !$omp& num_threads(OMPVV_NUM_THREADS_HOST)
    DO i = 1, N
       !$omp atomic update
       x = x + 1
    END DO
    !$omp end parallel do simd

    OMPVV_TEST_AND_SET_VERBOSE(errors, x .ne. N)

    test_simd_atomic = errors
  END FUNCTION test_simd_atomic
END PROGRAM test_parallel_for_simd_atomic
