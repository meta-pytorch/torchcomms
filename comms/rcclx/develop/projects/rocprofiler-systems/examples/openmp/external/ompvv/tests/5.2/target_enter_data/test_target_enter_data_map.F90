!===-- test_target_enter_data_map.F90 --===//
! 
! 
! OpenMP API Version 5.2 June 2022
! 
! Testing implict behavior of target enter 
! and exit data map construct
! with global arrays
!
! 'target enter data' generates a target task
! 'map' clause determines how list items are 
!   mapped to device, and default data-sharing
!   attribute is shared in data environment
!   of the target task. 
! 
!===-----------------------------------===//
#include "ompvv.F90"

#define N 1024
PROGRAM test_target_enter_data_map
        use iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none

        OMPVV_TEST_OFFLOADING

        OMPVV_TEST_VERBOSE(test_enter_map() .ne. 0)

        OMPVV_REPORT_AND_RETURN()
CONTAINS
        INTEGER FUNCTION test_enter_map()
                INTEGER :: i
                INTEGER :: errors
                INTEGER, DIMENSION(10) :: A
                INTEGER, DIMENSION(10) :: B
                INTEGER, DIMENSION(10) :: D
                errors = 0
                do i = 1, 10
                        A(i) = 10
                        B(i) = 50
                        D(i) = 0
                end do
                !$omp target enter data map(A, B, D)
                        !$omp target map(tofrom: errors)
                                do i = 1, 10
                                        IF (A(i) /= 10) THEN
                                                errors = errors + 1
                                        END IF
                                        IF (B(i) /= 50) THEN
                                                errors = errors + 1
                                        END IF
                                        IF (D(i) /= 0) THEN
                                                errors = errors + 1
                                        END IF
                                 end do
                                 do i = 1, 10
                                        A(i) = 1000
                                        B(i) = 120
                                        D(i) = 70
                                 end do
                        !$omp end target
                !$omp target exit data map(A, B, D)
                do i = 1,10
                        OMPVV_TEST_AND_SET_VERBOSE(errors, B(i) /= 120)
                        OMPVV_TEST_AND_SET_VERBOSE(errors, A(i) /= 1000)
                        OMPVV_TEST_AND_SET_VERBOSE(errors, D(i) /= 70)
                end do
                test_enter_map = errors
        END FUNCTION test_enter_map
END PROGRAM test_target_enter_data_map
