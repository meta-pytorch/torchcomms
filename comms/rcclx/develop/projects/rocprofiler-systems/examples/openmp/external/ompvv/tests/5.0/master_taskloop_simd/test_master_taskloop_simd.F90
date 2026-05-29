!/===--- test_master_taskloop_simd.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the master taskloop simd directive in a parallel region.
! The test performs simple operations on an int array which are then checked
! for correctness.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_master_taskloop_simd
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(master_taskloop_simd() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION master_taskloop_simd()
    INTEGER:: errors = 0
    INTEGER, DIMENSION(N):: x, y, z
    INTEGER:: i
    INTEGER:: num_threads = -1

    OMPVV_INFOMSG("test_master_taskloop_simd")

    DO i = 1, N
       x(i) = 1
       y(i) = i + 1
       z(i) = 2*(i + 1)
    END DO

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z, num_threads) 
    !$omp master taskloop simd
    DO i = 1, N
       x(i) = x(i) + y(i) * z(i)
    END DO
    !$omp end master taskloop simd
    IF (omp_get_thread_num() .eq. 0) THEN
       num_threads = omp_get_num_threads()
    ENDIF
    !$omp end parallel

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. 1 + y(i)*z(i))
    END DO

    OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .lt. 1, "Test returned an invalid number of threads.")
    OMPVV_INFOMSG("This test does not guarantee vector instructions were generated for the simd construct.")

    master_taskloop_simd = errors
  END FUNCTION master_taskloop_simd
END PROGRAM test_master_taskloop_simd
