!/===--- test_parallel_master.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the parallel master directive. The test performs simple
! operations on an int array which are then checked for correctness.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_parallel_master
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(parallel_master() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION parallel_master()
    INTEGER:: errors = 0
    INTEGER:: num_threads = -1
    INTEGER, DIMENSION(N):: x, y, z
    INTEGER:: i

    OMPVV_INFOMSG("test_parallel_master")

    DO i = 1, N
       x(i) = 1
       y(i) = i + 1
       z(i) = 2*(i + 1)
    END DO

    !$omp parallel master num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z, num_threads) 
    !$omp taskloop
    DO i = 1, N
       x(i) = x(i) + y(i) * z(i)
    END DO
    !$omp end taskloop
    num_threads = omp_get_num_threads()
    !$omp end parallel master

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. 1 + y(i)*z(i))
    END DO

    OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .lt. 1, "Test returned an invalid number of threads.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .lt. 1)

    parallel_master = errors
  END FUNCTION parallel_master
END PROGRAM test_parallel_master
