!/===--- test_parallel_master_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the parallel master directive in a target context. The
! test performs simple operations on an int array which are then checked for
! correctness.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_parallel_master_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(parallel_master_device() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION parallel_master_device()
    INTEGER:: errors = 0
    INTEGER:: num_threads = -1
    INTEGER, DIMENSION(N):: x, y, z
    INTEGER:: i

    OMPVV_INFOMSG("test_parallel_master_device")

    DO i = 1, N
       x(i) = 1
       y(i) = i + 1
       z(i) = 2*(i + 1)
    END DO

    !$omp target map(tofrom: x, num_threads) map(to: y, z)
    !$omp parallel master num_threads(OMPVV_NUM_THREADS_DEVICE) shared(x, y, z, num_threads) 
    !$omp taskloop
    DO i = 1, N
       x(i) = x(i) + y(i) * z(i)
    END DO
    !$omp end taskloop
    num_threads = omp_get_num_threads()
    !$omp end parallel master
    !$omp end target 

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. 1 + y(i)*z(i))
    END DO

    OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .lt. 1, "Test returned an invalid number of threads.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .lt. 1)

    parallel_master_device = errors
  END FUNCTION parallel_master_device
END PROGRAM test_parallel_master_device
