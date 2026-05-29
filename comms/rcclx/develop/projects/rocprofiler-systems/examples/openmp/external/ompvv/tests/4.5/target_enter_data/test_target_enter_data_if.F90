!===---- test_target_enter_data.F90 - checking for the if clause of target enter data ------===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! The if clause determines if the section should be executed in the host or 
! the device. There are three things to test here: 
! (a) with offloading when 'if' clause evaluates to true then code
! be executed on the device 
! (b) with offloading when 'if' clause evaluates to false then code should
! be executed on the host
! (c) without offloading all the code should be executed on the device
! The if clause is evaluated on runtime which means that variables could
! determine this behavior. We use a SIZE_THRESHOLD variable to check if we 
! should execute on the device or the host. Before starting the test we 
! sample offloading to see if it was enabled or not. If the code is executed
! in the device, the result should be c(i) = a(i) + b(i) = i + 1. 
! If the code is executed on the host the result should be c(i) = -1
!
!===----------------------------------------------------------------------===//

#include "ompvv.F90" 

#define SIZE_ARRAY 1024
#define THRESHOLD 512

      PROGRAM tests_target_enter_data_if
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
       
         LOGICAL :: isOffloading, isSharedEnv
         INTEGER :: a(1:SIZE_ARRAY)
         INTEGER :: b(1:SIZE_ARRAY)
         INTEGER :: c(1:SIZE_ARRAY)
         INTEGER :: i, j, s
         CHARACTER(len=300) :: infoMessage
        
         OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
         OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)
         IF (isSharedEnv) THEN 
           WRITE(infoMessage, *) " working on a shared memory &
           & environment. Testing data mapping not possible"
           OMPVV_WARNING(infoMessage)
           OMPVV_REPORT_AND_RETURN()
         END IF
         IF (.NOT. isOffloading) THEN 
           WRITE(infoMessage, *) "Offloading is off. Might not be&
           & possible to test if clause correctly "
           OMPVV_WARNING(infoMessage)
          OMPVV_REPORT_AND_RETURN()
         END IF

         ! We only test if it is offloading and it is not shared 
         ! memory. As wiehtout these two conditions would not 
         ! be possible to test data transfers
         OMPVV_TEST_VERBOSE(test_target_enter_data_if() .NE. 0) 
         
         OMPVV_REPORT_AND_RETURN()

       CONTAINS 
         INTEGER FUNCTION test_target_enter_data_if()
           ! check multiple sizes. 
           DO s = 256, SIZE_ARRAY, 256
             ! a, b and c arrays initialization
             a(:) = (/ (SIZE_ARRAY - i, i = 1, SIZE_ARRAY) /)
             b(:) = (/ (i - 1, i = 1, SIZE_ARRAY) /) 
             c(:) = -1
             !$omp target enter data if(s > THRESHOLD) &
             !$omp map(to: a(1:s), b(1:s)) 

             ! change the value at the host
             a(:) = 0
             b(:) = 0
                               
             ! If the target enter data did not map, then the 
             ! map(to...) will get the updated value, but if it did
             ! map then the map(to...) should be a no_op

             !$omp target map(to: a(1:s), b(1:s)) &
             !$omp map(tofrom: c(1:s)) map(tofrom: s)

               c(1:s) = (/ (a(j) + b(j) + 1, j = 1, s) /)

             !$omp end target

             ! Checking results
             IF (s > THRESHOLD) THEN
               ! check for all values that should have been mapped 
               ! on the target enter data
               OMPVV_TEST_VERBOSE(ANY(c(:s) /= SIZE_ARRAY))
             ELSE
               ! checking for values that should have been 
               ! mapped by the target region and not the target enter
               ! data
               OMPVV_TEST_VERBOSE(ANY(c(:s) /= 1))
               WRITE(infoMessage, *) "s = ", s
             END IF 

             ! This is not part of the test but it is necessary for garbage
             ! collection
             !$omp target exit data if(s > THRESHOLD) map(delete: a(1:s), b(1:s)) 

           END DO ! s

           OMPVV_GET_ERRORS(test_target_enter_data_if)
         END FUNCTION test_target_enter_data_if
           
      END PROGRAM tests_target_enter_data_if
