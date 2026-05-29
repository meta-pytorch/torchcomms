!===-test_ompx.F-===/
!
! OpenMP API Version 5.2 July 2022
!
! Testing 'omx' sentinel
!
! 'omx' sentinel is reserved for implementation-defined extensions to
! fixed source form OpenMP directives.
! TODO: this program uses ompvv_lib form ompvv.F, but it was not
! possible to use the macros defined there and used in most tests of the
! testsuit. The macros were left as commands in case in the future
! compilers of the ompvv.F is fixed to make the macros usable.
! 
!===-------------------------------===//
#include "ompvv.F"

#define N 1024
      PROGRAM test_omx
        use iso_fortran_env
        use ompvv_lib
        use omp_lib
        implicit none
        INTEGER res
        LOGICAL t


        !OMPVV_TEST_OFFLOADING
        ! Not using the macros, since tested compilers
        ! do not support them
        ! __FILE__ does not work
        call test_offloading("test_ompx.F", __LINE__)

        res = test_fixed_omx()
        t = res .NE. 2
        !OMPVV_TEST_VERBOSE(res .NE. 2)
        call test_error_verbose(t,"condition","test_ompx.F",__LINE__)

        !OMPVV_REPORT_AND_RETURN()
        res = report_and_set_errors("test_ompx.F")
        IF( res .NE. 0) THEN
          stop EXIT_FAILURE
        ENDIF
      CONTAINS
        INTEGER FUNCTION test_fixed_omx() 
          INTEGER :: i
          INTEGER :: n
          INTEGER :: errors
          INTEGER, DIMENSION(2) :: ARR_ERR
          errors = 0
!$omp parallel shared(ARR_ERR) private(i)  num_threads(2)
!$omx test_nonexistant_omx
          i = omp_get_thread_num()
          i = i + 1
          ARR_ERR(i) = 1
!$omp end parallel
          do n = 1,2
            errors = errors + ARR_ERR(n)
          end do
          test_fixed_omx = errors
        END FUNCTION test_fixed_omx
      END PROGRAM test_omx
