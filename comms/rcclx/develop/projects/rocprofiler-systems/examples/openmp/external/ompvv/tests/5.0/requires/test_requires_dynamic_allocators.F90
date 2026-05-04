!//===------ test_requires_dynamic_allocators.F90 --------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Tests the requires dynamic allocators clause by working with an OMP
! allocator inside a target region. Allocators testing is based on the
! OpenMP 5.0 example for allocators. The allocator testing first creates
! an allocator, with 64-byte alignment and the default memory space,
! then checks that 64-byte alignment is correct and that the memory can
! be written to in the target region. The tests checks that the values
! were written correctly, and then frees the memory and deletes the
! allocator.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_requires_dynamic_allocators
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  !$omp requires dynamic_allocators

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_dynamic_allocators() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_dynamic_allocators()
    INTEGER:: errors = 0
    INTEGER,ALLOCATABLE:: x(:)
    INTEGER:: x_save(N)
    INTEGER:: i
    INTEGER(omp_memspace_handle_kind):: x_memspace = omp_default_mem_space
    type(omp_alloctrait):: x_traits(1) = [omp_alloctrait(omp_atk_alignment,64)]
    INTEGER(omp_allocator_handle_kind):: x_alloc

    !$omp target defaultmap(tofrom)
    x_alloc = omp_init_allocator(x_memspace, 1, x_traits)

    !$omp allocate(x) allocator(x_alloc)
    allocate(x(N))

    !$omp parallel
    !$omp do simd simdlen(16) aligned(x: 64)
    DO i = 1, N
       x(i) = i
    END DO
    !$omp end parallel

    DO i = 1, N
       x_save(i) = x(i)
    END DO

    deallocate(x)
    call omp_destroy_allocator(x_alloc)

    !$omp end target

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, x_save(i) .ne. i)
    END DO


    test_dynamic_allocators = errors
  END FUNCTION test_dynamic_allocators
END PROGRAM test_requires_dynamic_allocators
