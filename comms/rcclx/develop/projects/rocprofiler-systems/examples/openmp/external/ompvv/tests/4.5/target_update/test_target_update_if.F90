!===-- test_target_update_if.F90 --------------------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! The if clause determines if the section should be updated on  
! the device. There are two scenarios to test here: 
! (a) with offloading when 'if' clause evaluates to true then 
! associated data is updated depending on the motion clause.
! (b) with offloading when 'if' clause evaluates to false 
! then there is no update
! The if clause is evaluated on runtime which means that variables could
! determine this behavior.
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_update_if
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(target_update_if() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_update_if()
      INTEGER :: i, j, k, counts, toggle, report_errors
      INTEGER, DIMENSION(N) :: a, b, c
      INTEGER, DIMENSION(2) :: errors
      LOGICAL :: is_offloading, change_flag
      toggle = 0 
      i = 0
      j = 0
      k = 0
      errors = 0
      report_errors = 0
      change_flag = .FALSE.
      is_offloading = .FALSE.
      
      DO i = 1, N 
         a(i) = 10
      END DO

      OMPVV_TEST_AND_SET_OFFLOADING(is_offloading)
     
      IF (is_offloading .neqv. .TRUE.) THEN
         OMPVV_WARNING("It is not possible to test conditional data transfers if the environment is shared or offloading is off")
      END IF
      
      DO counts = 1, 4
         DO i = 1, N
            b(i) = 2
            c(i) = 0
         END DO

         !$omp target data map(to: a, b) map(tofrom: c)
            !$omp target
               DO j = 1, N
                  c(j) = a(j) + b(j)
               END DO
            !$omp end target
         !$omp end target data  
            
         change_flag = init_b()

         !$omp target update if(change_flag) to(b(:N))

         !$omp target
            DO k = 1, N
               c(k) = c(k) + b(k)   
            END DO 
         !$omp end target

         IF (change_flag .eqv. .TRUE.) THEN
            DO i = 1, N
               IF (c(i) .ne. 16) THEN
                  errors(1) = errors(1) + 1
               END IF
            END DO
         ELSE
            DO i = 1, N
               IF (c(i) .ne. 14) THEN
                  errors(2) = errors(2) + 1
               END IF
            END DO
         END IF
      
      END DO
   
      OMPVV_TEST_AND_SET_VERBOSE(report_errors, errors(1) > 0)
      OMPVV_INFOMSG_IF(errors(1) > 0, "Target update test when if clause is true failed")
      OMPVV_TEST_AND_SET_VERBOSE(report_errors, errors(2) > 0)
      OMPVV_INFOMSG_IF(errors(2) > 0, "Target update test when if clause is false failed")
   
   target_update_if = report_errors
   END FUNCTION target_update_if

   LOGICAL FUNCTION init_b()
      INTEGER :: i, toggle
      INTEGER, DIMENSION(N) :: b
      LOGICAL :: change_flag
      toggle = 0
 
      IF (MODULO(toggle, 2) .ne. 0) THEN
         DO i = 1, N
            b(i) = b(i) * 2
         END DO
         toggle = toggle + 1
         change_flag = .TRUE.
      ELSE
         toggle = toggle + 1
         change_flag = .FALSE.
      END IF
      init_b = change_flag
   END FUNCTION init_b

END PROGRAM test_target_update_if
