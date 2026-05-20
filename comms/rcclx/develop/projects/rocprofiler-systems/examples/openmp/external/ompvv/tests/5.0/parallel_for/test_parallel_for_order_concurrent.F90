!/===--- test_parallel_for_order_concurrent.F90 ---------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the parallel for directive with the order(concurrent)
! The test performs simple operations on an int array which are then
! checked for correctness. The specification indicates only that iterations
! in the loop may be executed concurrently, so no particular order or lack 
! thereof can be required by this test.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_parallel_for_order_concurrent
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(parallel_for_order_concurrent() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION parallel_for_order_concurrent()
    INTEGER:: errors = 0
    INTEGER:: i
    INTEGER,DIMENSION(N):: x, y, z

    OMPVV_INFOMSG("test_parallel_for_order_concurrent")

    DO i = 1, N
       x(i) = 1
       y(i) = i + 1
       z(i) = 2 * (i + 1)
    END DO

    !$omp parallel do order(concurrent) &
    !$omp& num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z)
    DO i = 1, N
       x(i) = x(i) + y(i) * z(i)
    END DO
    !$omp end parallel do

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. 1 + (y(i)*z(i)))
    END DO

    parallel_for_order_concurrent = errors
  END FUNCTION parallel_for_order_concurrent
END PROGRAM test_parallel_for_order_concurrent
