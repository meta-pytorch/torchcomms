!===-test_ompx.F90-===/
!
!
! OpenMP API Version 5.2 July 2022
!
! Testing 'ompx' sentinel
!
! 'ompx' sentinel is reserved for
! implementation-defined extensions to
! free source form OpenMP directives.
!===-------------------------------===//
#include "ompvv.F90"

#define N 1024
PROGRAM test_ompx
        use iso_fortran_env
        use ompvv_lib
        use omp_lib
        implicit none

        OMPVV_TEST_OFFLOADING

        OMPVV_TEST_VERBOSE(test_free_ompx() .ne. 2)

        OMPVV_REPORT_AND_RETURN()
CONTAINS
        INTEGER FUNCTION test_free_ompx() 
                INTEGER :: i
                INTEGER :: n
                INTEGER :: errors
                INTEGER, DIMENSION(2) :: ARR_ERR
                errors = 0
                !$omp parallel shared(ARR_ERR) private(i)  num_threads(2)
                !$ompx test_nonexistent
                        i = omp_get_thread_num()
                        i = i + 1
                        ARR_ERR(i) = 1
                !$omp end parallel
                do n = 1,2
                        errors = errors + ARR_ERR(n)
                end do
                test_free_ompx = errors
        END FUNCTION test_free_ompx
END PROGRAM test_ompx
