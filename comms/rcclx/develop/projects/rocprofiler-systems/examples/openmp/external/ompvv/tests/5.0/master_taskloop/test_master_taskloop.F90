!/===--- test_master_taskloop.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the master taskloop directive in a parallel region. The
! test performs simple operations on an int array which are then checked for
! correctness.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_master_taskloop
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(master_taskloop() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION master_taskloop()
    INTEGER:: errors = 0
    INTEGER:: num_threads = -1
    INTEGER, DIMENSION(N):: x, y, z
    INTEGER:: i

    OMPVV_INFOMSG("test_master_taskloop")

    DO i = 1, N
       x(i) = 1
       y(i) = i + 1
       z(i) = 2*(i + 1)
    END DO

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z, num_threads) 
    !$omp master taskloop
    DO i = 1, N
       x(i) = x(i) + y(i) * z(i)
    END DO
    !$omp end master taskloop
    IF (omp_get_thread_num() .eq. 0) THEN
       num_threads = omp_get_num_threads()
    END IF
    !$omp end parallel

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. 1 + y(i)*z(i))
    END DO

    OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread, so parallelism of taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .lt. 1, "Test returned an invalid number of threads.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .lt. 1)

    master_taskloop = errors
  END FUNCTION master_taskloop
END PROGRAM test_master_taskloop
