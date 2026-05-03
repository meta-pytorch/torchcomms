!===--- test_target_enter_exit_data_depend.F90 ------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
! 
! This test checks functionality of target enter data and target exit
! data to depend 'in' and 'out' using two separate functions. The first
! function test_async_between_task_target() mixes host-based tasks with
! target-based tasks, while the second function test_async_between_target() 
! is testing for target enter exit data to depend 'in' and 'out' respectively,
! while also checking that a nowait clause can be used to ensure asynchronous
! behavior. 
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_enter_exit_data_depend
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_OFFLOADING

   OMPVV_TEST_VERBOSE(test_async_between_task_target() .ne. 0)
   OMPVV_TEST_VERBOSE(test_async_between_target() .ne. 0) 
   
   OMPVV_REPORT_AND_RETURN()

   CONTAINS 
      INTEGER FUNCTION test_async_between_task_target()
         INTEGER :: errors, i
         REAL :: summation
         REAL, TARGET, DIMENSION(N) :: compute_array, in_1, in_2
         REAL, POINTER, DIMENSION(:) :: h_array, in_1_ptr, in_2_ptr
         compute_array(:) = 0
         in_1(:) = 0
         in_2(:) = 0
         h_array => compute_array
         in_1_ptr => in_1
         in_2_ptr => in_2
        
         errors = 0
         summation = 0.0
         !$omp task depend(out: in_1_ptr) shared(in_1_ptr)
            DO i = 1, N
               in_1_ptr(i) = 1
            END DO
         !$omp end task

         !$omp task depend(out: in_2_ptr) shared(in_2_ptr)
            DO i = 1, N
               in_2_ptr(i) = 2
            END DO
         !$omp end task

         !$omp target enter data map(alloc: h_array) map(to: in_1_ptr)&
         !$omp& map(to: in_2_ptr) depend(out: h_array) depend(in: in_1_ptr) depend(in: in_2_ptr)
         
         !$omp task shared (h_array, in_1_ptr, in_2_ptr) depend(inout: h_array) depend(in: in_1_ptr) depend(in: in_2_ptr)
            !$omp target
               DO i = 1, N
                  h_array(i) = in_1_ptr(i) * in_2_ptr(i)
               END DO
            !$omp end target
         !$omp end task
 
         !$omp target exit data map(from: h_array) depend(inout: h_array)
 
         !$omp task depend(in: h_array) shared(summation, h_array)
            !---checking results---!
            DO i = 1, N 
               summation = summation + compute_array(i)
            END DO
         !$omp end task
         
         !$omp taskwait

         !$omp target exit data map(release: h_array) map(release: in_1_ptr)&
         !$omp& map(release: in_2_ptr)

         OMPVV_TEST_AND_SET(errors, 2.0*N .ne. summation)

         test_async_between_task_target = errors
      END FUNCTION test_async_between_task_target

      INTEGER FUNCTION test_async_between_target()
         INTEGER :: errors, i, summation, val
         INTEGER, TARGET, DIMENSION(N) :: compute_array
         INTEGER, POINTER, DIMENSION(:) :: h_array2
         compute_array(:) = 0
         h_array2 => compute_array
         errors = 0
         summation = 0
         val = 2

         !$omp target enter data map(alloc: h_array2) depend(out: h_array2)

         !$omp target enter data map(to: val) depend(out: val)
 
         !$omp target depend(inout: h_array2) depend(in: val)
            DO i = 1, N
               h_array2(i) = val
            END DO
         !$omp end target
         
         !$omp target exit data map(from: h_array2) depend(in: h_array2)

         !$omp taskwait
           
         !$omp target exit data map(release: val)

         !---checking results---!
         DO i = 1, N
            summation = summation + compute_array(i)
         END DO

         OMPVV_TEST_AND_SET(errors, 2*N .ne. summation)

         test_async_between_target = errors
      
      END FUNCTION test_async_between_target
         
END PROGRAM test_target_enter_exit_data_depend
