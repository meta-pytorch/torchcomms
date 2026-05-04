!//===------ test_atomic_acquire_release.F90 --------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Adapted from OpenMP examples acquire_release.2.c
! When the atomic read operation on thread 1 reads a non-zero value from y,
! this results in a release/acquire synchronization that in turn implies that 
! the assignment to x on thread 0 happens before the read of x on thread 1. 
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1000

PROGRAM test_atomic_acquire_release_program
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_atomic_acquire_release() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_atomic_acquire_release()
    INTEGER:: x, y, thrd, tmp, errors

    x = 0
    y = 0
    errors = 0

    OMPVV_INFOMSG("test_atomic_acquire_release")
    
    call omp_set_dynamic(.false.)
    call omp_set_num_threads(2)

    !$omp parallel private(thrd, tmp)
       thrd = omp_get_thread_num()
       IF (thrd .EQ. 0) THEN
          x = 10 
          !$omp atomic write release
          y = 1
          !$omp end atomic
       ELSE
          tmp = 0
          DO WHILE (tmp .EQ. 0) 
            !$omp atomic read acquire
            tmp = y
            !$omp end atomic 
          END DO
          OMPVV_TEST_AND_SET_VERBOSE(errors, x .NE. 10)
       END IF
    !$omp end parallel

    test_atomic_acquire_release = errors
  END FUNCTION test_atomic_acquire_release
END PROGRAM test_atomic_acquire_release_program
