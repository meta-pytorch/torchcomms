!//===------ test_taskloop_in_reduction_device.F90 ------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the taskloop directive with the `in_reduction` reduction
! participation clause. It performs simple array operations which are added
! to a reduction variable in a taskloop with the in_reduction clause. This
! test checks the above in a target context.
!
!//===---------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_taskloop_in_reduction_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_taskloop_in_red_device() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_taskloop_in_red_device()
    INTEGER:: errors, test_sum, num_threads, expected_sum, i
    INTEGER, DIMENSION(N):: y, z

    OMPVV_INFOMSG("test_taskloop_in_reduction_device")

    errors = 0
    num_threads = -1
    test_sum = 0
    expected_sum = 0

    DO i = 1, N
       y(i) = i + 1
       z(i) = 2*(i + 1)
    END DO

    !$omp target parallel reduction(task, +:test_sum) num_threads(OMPVV_NUM_THREADS_DEVICE) shared(y, z, num_threads) defaultmap(tofrom)
       !$omp master
       !$omp taskloop in_reduction(+:test_sum)
       DO i = 1, N
          test_sum = test_sum + y(i)*z(i)
       END DO
       !$omp end taskloop
       num_threads = omp_get_num_threads()
       !$omp end master
    !$omp end target parallel

    DO i = 1, N
       expected_sum = expected_sum + y(i)*z(i)
    END DO

    OMPVV_TEST_AND_SET_VERBOSE(errors, test_sum .NE. expected_sum)

    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so parallelism of taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .LT. 1, "Test returned an invalid number of threads.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 1)

    test_taskloop_in_red_device = errors
  END FUNCTION test_taskloop_in_red_device
END PROGRAM test_taskloop_in_reduction_device
