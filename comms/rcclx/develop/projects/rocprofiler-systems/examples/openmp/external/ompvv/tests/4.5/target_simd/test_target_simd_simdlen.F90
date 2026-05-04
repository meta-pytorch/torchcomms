!===---- test_target_simd_simdlen.F90 - combined construct simd target -===//
! 
! OpenMP API Version 4.5 Nov 2015
!
! This test checks for the use of the simdlen clause which provides a hint to the
! compiler to the appropriate number of simd lanes when generating the simd
! region of code. However, different to safelen, this clause does not guarantee 
! that the number of lanes will actually be the one specified. The number of lanes
! is implementation defined 
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
            INTEGER, ALLOCATABLE :: A(:)
            INTEGER :: errors_bf, errors_af
            INTEGER :: x

            OMPVV_INFOMSG("test_target_simd")

            OMPVV_GET_ERRORS(errors_bf)
            ALLOCATE(A(ARRAY_SIZE))

            ! A and A_host initialization
            A(:) = 1

            ! Testing on the device the safelen for values 
            ! 1 5 8 13 16 100 128

            !$omp target simd safelen(1) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(5) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(8) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(13) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(16) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(100) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(128) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 1, ARRAY_SIZE
              A(x) = A(x) + A(x)
            END DO
            !$omp end target simd

            ! CHECKING THE RESULTS
            OMPVV_TEST_VERBOSE(ANY(A(:) .NE. 128))

            OMPVV_GET_ERRORS(errors_af)
            test_combined_target_simd_simple = errors_af - errors_bf
          END FUNCTION test_combined_target_simd_simple
      END PROGRAM test_target_simd

