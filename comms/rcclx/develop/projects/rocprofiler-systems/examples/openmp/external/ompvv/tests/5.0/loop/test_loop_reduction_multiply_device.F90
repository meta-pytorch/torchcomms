!===--- test_loop_reduction_multiply_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a loop directive, testing that the
! variable in the reduction clause is properly reduced using the multiply
! operator. This test checks the above in a target context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_loop_reduction_multiply_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_multiply() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_multiply()
    INTEGER,DIMENSION(N):: a, num_threads
    INTEGER:: errors, device_result, host_result, x, y
    CHARACTER(len=400) :: msgHelper
    INTEGER :: seedSize
    INTEGER,ALLOCATABLE :: seed(:)
    DOUBLE PRECISION:: randomNumber

    errors = 0
    CALL random_seed(size=seedSize)
    ALLOCATE(seed(seedSize))
    seed = 1
    CALL random_seed(put=seed)
    DEALLOCATE(seed)

    DO x = 1, N
       !random_number() generates a real number, r, uniformly distributed in 0 <= r < 1.
       CALL random_number(randomNumber)
       IF ( randomNumber .eq. 0.0 ) THEN
         a(x) = 2
       ELSE
         a(x) = 1
       ENDIF
       num_threads(x) = -1 * x
    END DO

    device_result = 1

    DO x = 1, N, 16
      device_result = 1
      !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: device_result, a, num_threads)
      !$omp loop reduction(*:device_result)
      DO y = 0, 15
         device_result = device_result * a(x + y)
      END DO
      !$omp end loop
      !$omp do
      DO y = 0, 15
         num_threads(x + y) = omp_get_num_threads()
      END DO
      !$omp end do
      !$omp end target parallel
      host_result = 1
      DO y = 0, 15
         host_result = host_result * a(x + y)
      END DO
      OMPVV_TEST_AND_SET_VERBOSE(errors, host_result .ne. device_result)
      WRITE(msgHelper, *) "Loop directive result is ", device_result, "and expected result is ", host_result, "."
      OMPVV_ERROR_IF(host_result .ne. device_result, msgHelper)
    END DO

    DO x = 2, N
       OMPVV_WARNING_IF(num_threads(x - 1) .ne. num_threads(x), "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.")
    END DO
    OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Test operated with one thread.  Reduction clause cannot be tested.")
    OMPVV_WARNING_IF(num_threads(1) .le. 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.")

    test_multiply = errors
  END FUNCTION test_multiply
END PROGRAM test_loop_reduction_multiply_device
