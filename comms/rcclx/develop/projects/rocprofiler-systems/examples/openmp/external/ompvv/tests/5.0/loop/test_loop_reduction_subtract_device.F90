!===--- test_loop_reduction_subtract_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a loop directive, testing that the
! variable in the reduction clause is properly reduced using the subtract
! operator. This test checks the above in a target context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_loop_reduction_subtract_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_subtraction() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_subtraction()
    INTEGER,DIMENSION(N):: a, b, num_threads
    INTEGER:: total, host_total, errors, x
    CHARACTER(len=400) :: msgHelper

    total = 0
    host_total = 0
    errors = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
       num_threads(x) = -1 * x
    END DO

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: total, a, b, num_threads)
    !$omp loop reduction(-:total)
    DO x = 1, N
       total = total - (a(x) + b(x))
    END DO
    !$omp end loop
    !$omp do
    DO x = 1, N
       num_threads(x) = omp_get_num_threads()
    END DO
    !$omp end do
    !$omp end target parallel

    DO x = 1, N
       host_total = host_total - (a(x) + b(x))
    END DO

    DO x = 2, N
       OMPVV_WARNING_IF(num_threads(x - 1) .ne. num_threads(x), "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.")
    END DO
    OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Test operated with one thread.  Reduction clause cannot be tested.")
    OMPVV_WARNING_IF(num_threads(1) .le. 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.")

    OMPVV_TEST_AND_SET_VERBOSE(errors, host_total .ne. total)
    WRITE(msgHelper, *) "Total from loop directive is ", total, " but expected total is ", host_total, "."
    OMPVV_ERROR_IF(host_total .ne. total, msgHelper)

    test_subtraction = errors
  END FUNCTION test_subtraction
END PROGRAM test_loop_reduction_subtract_device
