!===--- test_loop_reduction_max_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a loop directive, testing that the
! variable in the reduction clause is properly reduced using the max
! operator. This test checks the above in a target context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_loop_reduction_max_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_max() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_max()
    INTEGER,DIMENSION(N):: a, b, num_threads
    INTEGER:: errors, device_result, host_max, x
    CHARACTER(len=400) :: msgHelper
    INTEGER :: seedSize
    INTEGER,ALLOCATABLE :: seed(:)
    DOUBLE PRECISION:: randomNumber
    INTEGER:: randomInteger

    errors = 0
    CALL random_seed(size=seedSize)
    ALLOCATE(seed(seedSize))
    seed = 1
    CALL random_seed(put=seed)
    DEALLOCATE(seed)

    DO x = 1, N
       !random_number() generates a real number, r, uniformly distributed in 0 <= r < 1.
       CALL random_number(randomNumber)
       randomInteger = INT(randomNumber*100.0)
       a(x) = randomInteger
       CALL random_number(randomNumber)
       randomInteger = INT(randomNumber*100.0)
       b(x) = randomInteger
       num_threads(x) = -1 * x
    END DO

    device_result = 0

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: device_result, a, b, num_threads)
    !$omp loop reduction(max:device_result)
    DO x = 1, N
      device_result = MAX(a(x) + b(x), device_result)
    END DO
    !$omp end loop
    !$omp do
    DO x = 1, N
      num_threads(x) = omp_get_num_threads()
    END DO
    !$omp end do
    !$omp end target parallel

    host_max = 0
    DO x = 1, N 
      host_max = MAX(host_max, a(x) + b(x))
    END DO

    DO x = 2, N
       OMPVV_WARNING_IF(num_threads(x - 1) .ne. num_threads(x), "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.")
    END DO
    OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Test operated with one thread.  Reduction clause cannot be tested.")
    OMPVV_WARNING_IF(num_threads(1) .le. 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.")

    OMPVV_TEST_AND_SET_VERBOSE(errors, device_result .ne. host_max)
    WRITE(msgHelper, *) "Max from loop directive is ", device_result, " but expected max is ", host_max, "." 
    OMPVV_ERROR_IF(host_max .ne. device_result, msgHelper)

    test_max = errors
  END FUNCTION test_max
END PROGRAM test_loop_reduction_max_device
