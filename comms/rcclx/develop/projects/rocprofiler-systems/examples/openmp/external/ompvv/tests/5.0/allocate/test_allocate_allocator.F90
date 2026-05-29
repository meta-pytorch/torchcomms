!//===------ test_allocate_allocators.F90 ----------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Tests the allocate directive with allocator clause, based on the OpenMP
! 5.0 example for allocators. The allocator testing first creates
! an allocator, with 64-byte alignment and the default memory space,
! then checks that 64-byte alignment is correct and that the memory can
! be written to in the parallel region. The tests checks that the values
! were written correctly, and then frees the memory and deletes the
! allocator.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_allocate_allocators
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_allocators() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_allocators()
    INTEGER:: errors = 0
    INTEGER,ALLOCATABLE:: x(:)
    INTEGER:: i
    INTEGER(omp_memspace_handle_kind):: x_memspace = omp_default_mem_space
    type(omp_alloctrait):: x_traits(1) = [omp_alloctrait(omp_atk_alignment,64)]
    INTEGER(omp_allocator_handle_kind):: x_alloc

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
       OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .ne. i)
    END DO

    deallocate(x)
    call omp_destroy_allocator(x_alloc)

    test_allocators = errors
  END FUNCTION test_allocators
END PROGRAM test_allocate_allocators
