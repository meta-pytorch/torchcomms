!===-- test_fortran_pure_nothing.F90 --===//
! 
! 
! OpenMP API Version 5.2 November 2021
! 
! Testing new feature of nothing directive 
! in Fortran pure procedures. 
!
! The nothing directive should have
! no effect on an OpenMP Program. 
! 
! A Fortran pure procedure is a procedure
! without any chance of side effects outside
! of the procedure. 
!===-----------------------------------===// 

#include "ompvv.F90"
PROGRAM test_pure_nothing 
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none

        INTEGER:: errors
        INTEGER:: b
        errors = 0
        b = 1
        OMPVV_TEST_OFFLOADING

        OMPVV_TEST_VERBOSE(pure_nothing(b) .ne. 0)

        OMPVV_REPORT_AND_RETURN()
CONTAINS
        PURE INTEGER FUNCTION pure_nothing(a)
                !$omp nothing
                INTEGER, VALUE  :: a
                a = 0
                pure_nothing = a
        END FUNCTION pure_nothing
END PROGRAM test_pure_nothing        

