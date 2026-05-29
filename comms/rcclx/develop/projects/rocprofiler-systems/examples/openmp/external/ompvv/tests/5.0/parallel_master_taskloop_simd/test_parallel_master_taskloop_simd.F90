!/===--- test_parallel_master_taskloop_simd.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the parallel master taskloop simd directive. The
! test performs simple operations on an int array which are then
! checked for correctness.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_parallel_master_taskloop_simd
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(parallel_master_taskloop_simd() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION parallel_master_taskloop_simd()
    INTEGER:: errors = 0
    INTEGER, DIMENSION(N):: x, y, z
    INTEGER:: i
    INTEGER:: num_threads = -1

    OMPVV_INFOMSG("test_parallel_master_taskloop_simd")

    DO i = 1, N
       x(i) = 1
       y(i) = i + 1
       z(i) = 2*(i + 1)
    END DO

    !$omp parallel master taskloop simd num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z, num_threads) 
    DO i = 1, N
       x(i) = x(i) + y(i) * z(i)
       IF (i .eq. 1) THEN
          num_threads = omp_get_num_threads()
       ENDIF
    END DO
    !$omp end parallel master taskloop simd

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. 1 + y(i)*z(i))
    END DO

    OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .lt. 1, "Test returned an invalid number of threads.")
    OMPVV_INFOMSG("This test does not guarantee vector instructions were generated for the simd construct.")

    parallel_master_taskloop_simd= errors
  END FUNCTION parallel_master_taskloop_simd
END PROGRAM test_parallel_master_taskloop_simd
