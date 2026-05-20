!/===--- test_parallel_for_allocate.c -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Tests the parallel for directive with allocator clause, based on the
! OpenMP 5.0 example for allocators. The allocator testing first creates
! an allocator, with 64-byte alignment and the default memory space,
! then checks that 64-byte alignment is correct and that the memory can
! be written to in the parallel for region in private arrays set to
! allocate with the created allocator. The tests checks that the values
! were written correctly, and then frees the memory and deletes the
! allocator.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_parallel_for_allocate
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_for_allocate() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_for_allocate()
    INTEGER:: errors = 0
    INTEGER:: successful_alloc = 0
    INTEGER:: i, j

    INTEGER,ALLOCATABLE:: x(:)
    INTEGER,DIMENSION(N,N):: result_arr

    INTEGER(omp_memspace_handle_kind):: x_memspace = omp_default_mem_space
    type(omp_alloctrait):: x_traits(1) = [omp_alloctrait(omp_atk_alignment,64)]
    INTEGER(omp_allocator_handle_kind):: x_alloc

    DO i = 1, N
       DO j = 1, N
          result_arr(i,j) = -1
       END DO
    END DO

    x_alloc = omp_init_allocator(x_memspace, size(x_traits), x_traits)

    !$omp parallel do allocate(x_alloc: x) private(x) &
    !$omp& shared(result_arr) num_threads(OMPVV_NUM_THREADS_HOST)
    DO i = 1, N
       allocate(x(N))
       IF (allocated(x)) THEN
          !$omp simd simdlen(16) aligned(x: 64)
          DO j = 1, N
             x(j) = j*i
          END DO
          DO j = 1, N
             result_arr(i, j) = x(j)
          END DO
          deallocate(x)
          successful_alloc = successful_alloc + 1
       END IF
    END DO
    !$omp end parallel do

    OMPVV_ERROR_IF(successful_alloc .lt. 1, "Failed to allocate x")
    OMPVV_TEST_AND_SET_VERBOSE(errors, successful_alloc .lt. 1)

    DO i = 1, N
       DO j = 1, N
          OMPVV_TEST_AND_SET_VERBOSE(errors, result_arr(i, j) .ne. i*j)
       END DO
    END DO

    call omp_destroy_allocator(x_alloc)

    test_for_allocate = errors
  END FUNCTION test_for_allocate
END PROGRAM test_parallel_for_allocate
