!===------FILE_NAME.F90 ---------------------------------------------------===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! 
!===----------------------------------------------------------------------===//
#include "ompvv.F90"

      program test
      use iso_fortran_env
      use ompvv_lib
      implicit none
        LOGICAL:: myFalseVar = .false.
        LOGICAL:: myTrueVar = .true.
        CHARACTER(len=500):: myTmpChar
        INTEGER:: errors

        OMPVV_TEST_OFFLOADING

        OMPVV_INFOMSG("This is an OMPVV_INFOMSG")
        WRITE(myTmpChar,*) "To include parameters or long info messages&
        & you need to use WRITE first to form the phrase"
        OMPVV_INFOMSG(myTmpChar)
        WRITE(myTmpChar,*) "This is an OMPVV_INFOMSG with concat &
        & parameters myTrueVar = ", myTrueVar
        OMPVV_INFOMSG(myTmpChar)
        OMPVV_INFOMSG_IF(myTrueVar,"This is an OMPVV_INFOMSG_IF(.true.)")
        OMPVV_INFOMSG_IF(myFalseVar,"This should not show up")
        OMPVV_INFOMSG_IF(1==1,"This is an OMPVV_INFOMSG_IF 1==1")
        OMPVV_WARNING("This is an OMPVV_WARNING")
        OMPVV_WARNING_IF(myTrueVar,"This is an OMPVV_WARNING_IF(.true.)")
        OMPVV_WARNING_IF(myFalseVar,"This should not show up")
        OMPVV_WARNING_IF(1==1,"This is an OMPVV_WARNING_IF 1==1")
        OMPVV_ERROR("This is an OMPVV_ERROR")
        OMPVV_ERROR_IF(myTrueVar,"This is an OMPVV_ERROR_IF(.true.)")
        OMPVV_ERROR_IF(myFalseVar,"This should not show up")
        OMPVV_ERROR_IF(1==1,"This is an OMPVV_ERROR_IF 1==1")

        WRITE(myTmpChar,*) "errors is internally handle, but you can &
        & obtain and set  the value with these functions"
        OMPVV_INFOMSG(myTmpChar)

        errors = 0
        OMPVV_TEST_AND_SET(errors, 1 /= 0)
        OMPVV_TEST(1 /= 0)
        OMPVV_TEST_AND_SET_VERBOSE(errors, 1 /= 0) ! Condition to generate an error and display
        OMPVV_TEST_VERBOSE(1 /= 0)
        OMPVV_REPORT() ! This should display test fail

        ! get teh current value of errors
        OMPVV_GET_ERRORS(errors)
        ! reset the number of errors
        OMPVV_SET_ERRORS(0)

        OMPVV_TEST_AND_SET(errors, 1 == 0)
        OMPVV_TEST(1 == 0)
        OMPVV_TEST_AND_SET_VERBOSE(errors, 1 == 0) ! Condition to generate an error and display
        OMPVV_TEST_VERBOSE(1 == 0)
        OMPVV_REPORT() ! This should display test fail

        errors = test_function()

        OMPVV_REPORT_AND_RETURN()

      contains
        function test_function() 
          INTEGER:: test_function
          test_function = 1;
          !$omp target map(from: errors)
            test_function = 0;
          !$omp end target
        end function test_function 

      end program test

