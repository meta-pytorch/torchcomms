!===---- test_target_simd_collapse.F90 - combined construct simd target -===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! SIMD in OpenMP 4.5 does not have any API that allows us to confirm the creation 
! of SIMD lanes, nor the use of SIMD instructions in any architecture. Hence, our
! tests are limited in that they check that the expected result is created, but
! assume nothing in how they are mapped into a particular architecture
!
! This test checks if the combined construct target simd with collapse produces 
! valid results on a simple computation
!===--------------------------------------------------------------------------===//

#include "ompvv.F90"

#define ARRAY_SIZE 1024

      PROGRAM test_target_simd_collapse
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        OMPVV_TEST_OFFLOADING

        OMPVV_WARNING("This test cannot check if actual SIMD extension")
        OMPVV_WARNING("at the hardware level were used, or if the")
        OMPVV_WARNING("the generated code is different in any way");
        OMPVV_TEST_VERBOSE(test_combined_target_simd_collapse() .ne. 0)
        OMPVV_REPORT_AND_RETURN()

        CONTAINS
          INTEGER FUNCTION test_combined_target_simd_collapse()
            INTEGER, ALLOCATABLE :: a(:,:), b(:,:), c(:,:)
            INTEGER :: errors_af, errors_bf, i, j

            OMPVV_INFOMSG("test_target_simd_collapse")
            OMPVV_GET_ERRORS(errors_bf)

            ALLOCATE(a(ARRAY_SIZE, ARRAY_SIZE))
            ALLOCATE(b(ARRAY_SIZE, ARRAY_SIZE))
            ALLOCATE(c(ARRAY_SIZE, ARRAY_SIZE))

            ! a b and c array initialization
            do i=1,ARRAY_SIZE
              do j=1,ARRAY_SIZE
                a(i,j) = 0
                b(i,j) = 2
                c(i,j) = 7
              enddo
            enddo


            !$omp target simd map(to: b(1:ARRAY_SIZE,1:ARRAY_SIZE), c(1:ARRAY_SIZE,1:ARRAY_SIZE)) &
            !$omp& map(tofrom: a(1:ARRAY_SIZE,1:ARRAY_SIZE)) collapse(2)
            do i=1,ARRAY_SIZE
              do j=1,ARRAY_SIZE
                a(i,j) = a(i,j) + b(i,j) * c(i,j) 
              enddo
            enddo
            !$omp end target simd

            OMPVV_TEST_VERBOSE(ANY(a(:,:) .NE. 14))
            OMPVV_GET_ERRORS(errors_af)
            test_combined_target_simd_collapse = errors_af - errors_bf
          END FUNCTION test_combined_target_simd_collapse
      END PROGRAM test_target_simd_collapse

