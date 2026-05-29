!===---- test_combined_target_simd.F90 - combined construct simd target -===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! SIMD in OpenMP 4.5 does not have any API that allows us to confirm the creation 
! of SIMD lanes, nor the use of SIMD instructions in any architecture. Hence, our
! tests are limited in that they check that the expected result is created, but
! assume nothing in how they are mapped into a particular architecture
!
! This test checks if the combined construct target simd produces valid
! results on a simple computation
!===--------------------------------------------------------------------------===//

#include "ompvv.F90"

#define ARRAY_SIZE 1024

      PROGRAM test_target_simd
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        OMPVV_TEST_OFFLOADING

        OMPVV_WARNING("This test cannot check if actual SIMD extension")
        OMPVV_WARNING("at the hardware level were used, or if the")
        OMPVV_WARNING("the generated code is different in any way");
        OMPVV_TEST_VERBOSE(test_combined_target_simd_simple() .ne. 0)
        OMPVV_REPORT_AND_RETURN()

        CONTAINS
          INTEGER FUNCTION test_combined_target_simd_simple()
            INTEGER, ALLOCATABLE :: a(:), b(:), c(:)
            INTEGER :: errors_af, errors_bf, x

            OMPVV_INFOMSG("test_target_simd")
            OMPVV_GET_ERRORS(errors_bf)

            ALLOCATE(a(ARRAY_SIZE))
            ALLOCATE(b(ARRAY_SIZE))
            ALLOCATE(c(ARRAY_SIZE))

            ! a b and c array initialization
            a(:) = 1
            b(:) = (/( x, x=0,ARRAY_SIZE-1)/)
            c(:) = (/( 2*x, x=0,ARRAY_SIZE-1)/)

            !$omp target simd map(to: b(1:ARRAY_SIZE), c(1:ARRAY_SIZE)) &
            !$omp& map(tofrom: a(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              a(x) = a(x) + b(x) * c(x)
            END DO
            !$omp end target simd

            OMPVV_TEST_VERBOSE(ANY(a(:) .NE. (/( 1 + 2*x*x, x = 0, ARRAY_SIZE-1)/)))
            OMPVV_GET_ERRORS(errors_af)
            test_combined_target_simd_simple = errors_af - errors_bf
          END FUNCTION test_combined_target_simd_simple
      END PROGRAM test_target_simd

