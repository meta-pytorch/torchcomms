!===--- test_metadirective_nothing.F90 -------------------------------------===//
!
! OpenMP API Version 5.1 Nov 2020
!
! Test for nothing directive within metadirectives. Runs a variety of
! metadirectives that check if the nothing directive is properly rendered.
! Primarily tests based on the fact that no matter what 'when' clause is 
! rendered it should result in nothing, and thus no additional pragma should
! be created. Thus, the threads should remain unchanged through this process
! and the compiler should handle it properly. Handles both device and host
! based tests.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024

PROGRAM test_metadirective_nothing
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  OMPVV_TEST_OFFLOADING

  IF( omp_get_num_devices() .GT. 0 ) THEN
    OMPVV_TEST_VERBOSE(metadirectiveOnDevice() .NE. 0)
  ELSE
    OMPVV_TEST_VERBOSE(metadirectiveOnHost() .NE. 0)
  END IF

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION metadirectiveOnDevice()
    INTEGER :: errors, i
    INTEGER :: A(N)
    INTEGER :: max_num_threads_target, max_num_threads_parallel

    errors = 0
    max_num_threads_target = 0
    max_num_threads_parallel = 0

    DO i=1, N
        A(i) = 0
    END DO

    !$omp target map(tofrom: A, max_num_threads_target, max_num_threads_parallel)
    max_num_threads_target = omp_get_max_threads()
    ! We expect at least one of these when conditons to eval to true, thus having the nothing directive utilized
    !$omp begin metadirective &
    !$omp when( device={kind(nohost)}: nothing ) &
    !$omp when( device={arch("nvptx")}: nothing ) &
    !$omp when( implementation={vendor(amd)}: nothing ) &
    !$omp default( parallel do num_threads(max_num_threads_target+1) )
    DO i=1, N
        IF( omp_in_parallel() ) THEN
            A(i) = A(i) + 1
        END IF
        IF( i .EQ. 1 ) THEN
            max_num_threads_parallel = omp_get_max_threads()
        END IF
    END DO
    !$omp end metadirective
    !$omp end target

    DO i=1, N
        OMPVV_TEST_AND_SET(errors, A(i) .NE. 0)
    END DO
    OMPVV_TEST_AND_SET(errors, max_num_threads_parallel .NE. max_num_threads_target)

    OMPVV_INFOMSG("Test ran with a number of available devices greater than 0")
    OMPVV_INFOMSG_IF(max_num_threads_parallel .EQ. max_num_threads_target, "Test recognized device was of arch/vendor/kind nvidia, amd, or nohost")
    OMPVV_WARNING_IF(A(1) .EQ. 1, "Test could not recognize if device was of arch/vendor/kind nvidia, amd or, nohost, even though there are devices available.")

    metadirectiveOnDevice = errors
  END FUNCTION metadirectiveOnDevice

  INTEGER FUNCTION metadirectiveOnHost()
    INTEGER :: errors, i
    INTEGER :: A(N)
    INTEGER :: max_num_threads_initial, max_num_threads_loop

    errors = 0
    max_num_threads_initial = 0
    max_num_threads_loop = 0

    DO i=1, N
        A(i) = 0
    END DO

    max_num_threads_initial = omp_get_max_threads()
    ! We expect all of these when statements to eval to false, causing body of code to run using 'nothing' as the default pragma
    !$omp begin metadirective &
    !$omp when( device={kind(nohost)}: parallel do num_threads(max_num_threads_initial+1) ) &
    !$omp when( device={arch("nvptx")}: parallel do num_threads(max_num_threads_initial+1) ) &
    !$omp when( implementation={vendor(amd)}: parallel do num_threads(max_num_threads_initial+1) ) &
    !$omp default( nothing )
    DO i=1, N
        IF( omp_in_parallel() ) THEN
            A(i) = A(i) + 1
        END IF
        IF( i .EQ. 1 ) THEN
            max_num_threads_loop = omp_get_max_threads()
        END IF
    END DO
    !$omp end metadirective

    OMPVV_WARNING_IF(A(1) .EQ. 1, "Even though no devices were available the test recognized kind/arch equal to nohost or nvptx or amd")

    DO i=1, N
        OMPVV_TEST_AND_SET(errors, A(i) .NE. 0)
    END DO
    OMPVV_TEST_AND_SET(errors, max_num_threads_initial .NE. max_num_threads_loop)

    metadirectiveOnHost = errors
  END FUNCTION metadirectiveOnHost
END PROGRAM test_metadirective_nothing
