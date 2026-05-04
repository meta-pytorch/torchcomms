!===---- test_combined_target_safele.F90 - simd directive clause safelen-===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! This test checks for the use of the safelen clause which prevents parallelization
! over SIMD lanes that goes behond the contant value passed to the clause. This
! is due to th possible iteration dependecies usually larger than 1. Regardless of 
! the SIMD len in hardware, the passed parameter should allow any possitive integer
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
        OMPVV_TEST_VERBOSE(test_target_simd_safelen() .ne. 0)
        OMPVV_REPORT_AND_RETURN()

        CONTAINS
          INTEGER FUNCTION test_target_simd_safelen()
            INTEGER, ALLOCATABLE :: A(:), A_host(:)
            INTEGER :: x
            INTEGER :: errors_af, errors_bf 
            
            OMPVV_INFOMSG("test_target_simd")

            ALLOCATE(A(ARRAY_SIZE))
            ALLOCATE(A_host(ARRAY_SIZE))

            OMPVV_GET_ERRORS(errors_bf)
            ! A and A_host initialization
            A(:) = 1
            A_host(:) = 1

            ! Testing on the device the safelen for values 
            ! 1 5 8 13 16 100 128

            !$omp target simd safelen(1) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 2, ARRAY_SIZE
              A(x-1) = A(x-1) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(5) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 6, ARRAY_SIZE
              A(x-5) = A(x-5) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(8) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 9, ARRAY_SIZE
              A(x-8) = A(x-8) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(13) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 14, ARRAY_SIZE
              A(x-13) = A(x-13) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(16) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 17, ARRAY_SIZE
              A(x-16) = A(x-16) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(100) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 101, ARRAY_SIZE
              A(x-100) = A(x-100) + A(x)
            END DO
            !$omp end target simd

            !$omp target simd safelen(128) map(tofrom: A(1:ARRAY_SIZE))
            DO x = 129, ARRAY_SIZE
              A(x-128) = A(x-128) + A(x)
            END DO
            !$omp end target simd


            ! Calculating the exepcted result on the host
            DO x = 2, ARRAY_SIZE
              A_host(x-1) = A_host(x-1) + A_host(x)
            END DO
            DO x = 6, ARRAY_SIZE
              A_host(x-5) = A_host(x-5) + A_host(x)
            END DO
            DO x = 9, ARRAY_SIZE
              A_host(x-8) = A_host(x-8) + A_host(x)
            END DO
            DO x = 14, ARRAY_SIZE
              A_host(x-13) = A_host(x-13) + A_host(x)
            END DO
            DO x = 17, ARRAY_SIZE
              A_host(x-16) = A_host(x-16) + A_host(x)
            END DO
            DO x = 101, ARRAY_SIZE
              A_host(x-100) = A_host(x-100) + A_host(x)
            END DO
            DO x = 129, ARRAY_SIZE
              A_host(x-128) = A_host(x-128) + A_host(x)
            END DO

            ! CHECKING THE RESULTS
            OMPVV_TEST_VERBOSE(ANY(A(:) .NE. A_host(:)))
            OMPVV_GET_ERRORS(errors_af)
            test_target_simd_safelen = errors_af - errors_bf
          END FUNCTION test_target_simd_safelen
      END PROGRAM test_target_simd

