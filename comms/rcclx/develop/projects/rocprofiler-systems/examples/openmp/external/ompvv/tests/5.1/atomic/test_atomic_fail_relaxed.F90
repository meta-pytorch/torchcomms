!//===------ test_atomic_fail_relaxed.F90 --------------------------===//
!
! OpenMP API Version 5.1 Nov 2020
!
! Utilizes an compare w/ an atomic release to ensure the implicit
! setting of x=10. Restrictions on atomic fail state they must be used
! on an atomic compare.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_atomic_fail_relaxed_program
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_atomic_fail_relaxed() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_atomic_fail_relaxed()
    INTEGER:: errors, x, y
    INTEGER:: thrd

    errors = 0
    x = 0
    y = 0

    OMPVV_INFOMSG("test_atomic_fail_relaxed")

    !$omp parallel num_threads(2) private(thrd)
    thrd = omp_get_thread_num()
    IF( thrd .EQ. 0 ) THEN
      y = 1
      !$omp atomic write seq_cst 
      x = 10
      !$omp end atomic
    ELSE
      DO WHILE ( y .NE. 5 )
        !$omp atomic compare fail(relaxed)
        IF (y == 1) THEN
          y = 5
        END IF 
        !$omp end atomic
      END DO
      OMPVV_TEST_AND_SET(errors, x .NE. 10)
      OMPVV_TEST_AND_SET(errors, y .NE. 5)
    END IF
    !$omp end parallel
    
    test_atomic_fail_relaxed = errors
  END FUNCTION test_atomic_fail_relaxed
END PROGRAM test_atomic_fail_relaxed_program
