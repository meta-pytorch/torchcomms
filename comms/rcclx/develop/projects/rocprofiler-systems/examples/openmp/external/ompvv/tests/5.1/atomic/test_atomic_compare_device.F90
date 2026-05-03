!//===------ test_atomic_compare_device.F90 --------------------------===//
!
! OpenMP API Version 5.1 Nov 2020
!
! Adapted from OpenMP example video https://www.youtube.com/watch?v=iS6IG7nzCSo
! Creates an array with random numbers, and uses atomic compare to find the max,
! testing against non-parallel maximum.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 100

PROGRAM test_atomic_compare_program
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_atomic_compare() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_atomic_compare()
    INTEGER:: errors, pmax, smax, i
    INTEGER:: arr(N)
    INTEGER:: assume, oldval, newval
    REAL:: rN

    errors = 0
    pmax = 0
    smax = 0

    OMPVV_INFOMSG("test_atomic_compare_device")

    DO i=1, N
      call RANDOM_NUMBER(rN)
      arr(i) = INT(rN*1000) 
    END DO

    !Sets max through non-parallel methods
    DO i=1, N
      IF( arr(i) > smax ) THEN
        smax = arr(i)
      END IF
    END DO

    !Sets max using parallel for loop, using atomic to ensure max is correct
    !$omp target parallel do map(pmax) shared(pmax) private(oldval, assume, newval)
    DO i=1, N
      oldval = pmax 
      DO 
        assume = oldval
        IF( arr(i) .GT. assume ) THEN
            newval = arr(i) 
        ELSE
            newval = assume
        END IF
        !$omp atomic compare capture
        IF( pmax .EQ. assume ) THEN
          pmax = newval 
        ELSE
          oldval = pmax
        END IF
        !$omp end atomic
        IF( assume .EQ. oldval) THEN
          EXIT
        END IF
      END DO
    END DO
    !$omp end target parallel do
    
    OMPVV_TEST_AND_SET(errors, pmax .NE. smax)    

    test_atomic_compare = errors
  END FUNCTION test_atomic_compare
END PROGRAM test_atomic_compare_program
