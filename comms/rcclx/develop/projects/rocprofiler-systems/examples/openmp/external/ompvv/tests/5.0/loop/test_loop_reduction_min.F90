!===--- test_loop_reduction_min.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a loop directive, testing that the
! variable in the reduction clause is properly reduced using the min
! operator.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_loop_reduction_min
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_min() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_min()
    INTEGER,DIMENSION(N):: a, b, num_threads
    INTEGER:: errors, device_result, host_min, x
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

    device_result = a(1) + b(1)

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
    !$omp loop reduction(min:device_result)
    DO x = 1, N
      device_result = MIN(a(x) + b(x), device_result)
    END DO
    !$omp end loop
    !$omp do
    DO x = 1, N
      num_threads(x) = omp_get_num_threads()
    END DO
    !$omp end do
    !$omp end parallel

    host_min = a(1) + b(1)
    DO x = 1, N 
      host_min = MIN(host_min, a(x) + b(x))
    END DO

    DO x = 2, N
       OMPVV_WARNING_IF(num_threads(x - 1) .ne. num_threads(x), "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.")
    END DO
    OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Test operated with one thread.  Reduction clause cannot be tested.")
    OMPVV_WARNING_IF(num_threads(1) .le. 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.")

    OMPVV_TEST_AND_SET_VERBOSE(errors, device_result .ne. host_min)
    WRITE(msgHelper, *) "Min from loop directive is ", device_result, " but expected min is ", host_min, "." 
    OMPVV_ERROR_IF(host_min .ne. device_result, msgHelper)

    test_min = errors
  END FUNCTION test_min
END PROGRAM test_loop_reduction_min
