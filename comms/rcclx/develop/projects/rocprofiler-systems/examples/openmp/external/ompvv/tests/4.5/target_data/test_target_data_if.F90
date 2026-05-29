!===---- test_target_if.F90 -  ------------------------------------------===//
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

      PROGRAM test_target_if
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none

         LOGICAL :: isOffloading, isHost
         INTEGER :: a(1:SIZE_ARRAY)
         INTEGER :: b(1:SIZE_ARRAY)
         INTEGER :: c(1:SIZE_ARRAY)
         INTEGER :: alpha, errors(3), i, j, s
         CHARACTER(len=300) :: infoMessage

         OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
         IF (.NOT. isOffloading) THEN 
           WRITE(infoMessage, *) "Offloading is off. Might not be&
           & possible to test if clause correctly "
           OMPVV_WARNING(infoMessage)
         END IF
         OMPVV_TEST_VERBOSE(tests_target_data_if_simple() .NE. 0)
         OMPVV_TEST_VERBOSE(tests_target_data_if_nested() .NE. 0)

         OMPVV_REPORT_AND_RETURN()

       CONTAINS
         INTEGER FUNCTION tests_target_data_if_simple()
           errors(:) = 0
           ! check multiple sizes.
           DO s = 256, 1024, 256
             ! a, b and c arrays initialization
             a(:) = (/ (SIZE_ARRAY - i, i = 1, SIZE_ARRAY) /)
             b(:) = (/ (i - 1, i = 1, SIZE_ARRAY) /)
             c(:) = -1
             !$omp target data if(s > THRESHOLD) &
             !$omp map(to: a(1:s), b(1:s)) &
             !$omp map(tofrom: c(1:s)) map(tofrom: isHost, s)

               ! If the condition is false, data mapping should not
               ! occur. We need to avoid a tofrom default mapping on
               ! a, b and c, we use alloc
               !$omp target map(alloc: a(1:s), b(1:s), c(1:s)) &
               !$omp map(tofrom: isHost, s)

                 isHost = omp_is_initial_device()
                 c(1:s) = (/ (a(j) + b(j) + 1, j = 1, s) /)

                 ! In case we have alloc, change the mem
                 ! content to avoid reading prev
                 ! values
                 a(1:s) = -1
                 b(1:s) = -1
               !$omp end target

               ! Should not exec in host if isOffloading is true
               IF (isOffloading) THEN
                 OMPVV_TEST_AND_SET_VERBOSE(errors(1), isHost)
               END IF
             !$omp end target data

             ! Checking data mapping

             DO i = 1, s
               IF (s > THRESHOLD .OR. &
                  & .NOT. isOffloading) THEN
                 ! Should have done data maping of a, b, and c
                 OMPVV_TEST_AND_SET(errors(2), (c(i) /= SIZE_ARRAY))
               ELSE
                 ! Should have not done data mapping of a, b or c
                 OMPVV_TEST_AND_SET(errors(3), (c(i) /= -1))
               END IF
             END DO ! i
           END DO ! s

           IF (errors(1) /= 0) THEN
             infoMessage = "Test did not offload to the device. if &
             & clause might be affecting the target offloading as well &
             & and it should not"
             OMPVV_ERROR(infoMessage)
           END IF
           infoMessage = MERGE("enabled ", "disabled", isOffloading)
           IF (errors(1) == 0 .AND. &
               & errors(2) == 0 .AND. &
               & errors(3) == 0) THEN
             infoMessage = "Test passed with offloading "//infoMessage
             OMPVV_INFOMSG(infoMessage)
           ELSE IF (errors(2) /= 0 .AND. errors(3) == 0) THEN
             infoMessage = "Test failed for if(.true.) &
             & with offloading "//infoMessage
             OMPVV_ERROR(infoMessage)
           ELSE IF (errors(2) == 0 .AND. errors(3) /= 0) THEN
             infoMessage = "Test failed for if(.false.) &
             & with offloading "//infoMessage
             OMPVV_ERROR(infoMessage)
           ELSE IF (errors(2) /= 0 .AND. errors(3) /= 0) THEN
             infoMessage = "Test fail for both cases if(.true.) and &
               & if(.false.) Incorrect data mapping &
               & with offloading "//infoMessage
             OMPVV_ERROR(infoMessage)
           END IF
           tests_target_data_if_simple = SUM(errors)
         END FUNCTION tests_target_data_if_simple

         INTEGER FUNCTION tests_target_data_if_nested()
           errors(:) = 0
           ! check multiple sizes.
           DO s = 256, 1024, 256
             ! a, b and c arrays initialization
             a(:) = (/ (SIZE_ARRAY - i, i = 1, SIZE_ARRAY) /)
             b(:) = (/ (i - 1, i = 1, SIZE_ARRAY) /)
             c(:) = -10
             !$omp target data if(s > THRESHOLD) &
             !$omp map(to: a(1:s), b(1:s)) &
             !$omp map(tofrom: c(1:s))

               !$omp target if(s > THRESHOLD) &
               !$omp map(alloc: a(1:s), b(1:s), c(1:s)) &
               !$omp map(tofrom: isHost, s)
                 isHost = omp_is_initial_device()
                 alpha = 0
                 IF (isHost) alpha = SIZE_ARRAY + 1
                 c(1:s) = (/ (a(j) + b(j) + 1 - alpha, j = 1, s) /)

                 ! In case we have alloc, change the mem
                 ! content to avoid reading prev
                 ! values
                 a(1:s) = -1
                 b(1:s) = -1
               !$omp end target

               ! Should not exec in host if isOffloading is true
               IF (isOffloading) THEN
                 IF (s > THRESHOLD) THEN
                   OMPVV_TEST_AND_SET_VERBOSE(errors(1), isHost )
                 ELSE
                   OMPVV_TEST_AND_SET_VERBOSE(errors(1), (.NOT. isHost))
                 END IF
               END IF
             !$omp end target data

             ! Checking data mapping

             DO i = 1, s
               IF (s > THRESHOLD .AND. isOffloading) THEN
                 ! Should have done data maping of a, b, and c
                 OMPVV_TEST_AND_SET(errors(2), (c(i) /= SIZE_ARRAY))
               ELSE
                 ! Should have not done data mapping of a, b or c
                 OMPVV_TEST_AND_SET(errors(3), (c(i) /= -1))
               END IF
             END DO ! i
           END DO ! s

           IF (errors(1) /= 0) THEN
             infoMessage = "IF statement inside of the inner target&
             & region did not handled offloading correctly"
             OMPVV_ERROR(infoMessage)
           END IF
           infoMessage = MERGE("enabled ", "disabled", isOffloading)
           IF (errors(1) == 0 .AND. &
               & errors(2) == 0 .AND. &
               & errors(3) == 0) THEN
             infoMessage = "Test passed with offloading "//infoMessage
             OMPVV_INFOMSG(infoMessage)
           ELSE IF (errors(2) /= 0 .AND. errors(3) == 0) THEN
             infoMessage = "Test failed for if(.true.) &
             & with offloading "//infoMessage
             OMPVV_ERROR(infoMessage)
           ELSE IF (errors(2) == 0 .AND. errors(3) /= 0) THEN
             infoMessage = "Test failed for if(.false.) &
             & with offloading "//infoMessage
             OMPVV_ERROR(infoMessage)
           ELSE IF (errors(2) /= 0 .AND. errors(3) /= 0) THEN
             infoMessage = "Test fail for both cases if(.true.) and &
               & if(.false.) Incorrect data mapping &
               & with offloading "//infoMessage
             OMPVV_ERROR(infoMessage)
           END IF
           tests_target_data_if_nested = SUM(errors)
         END FUNCTION tests_target_data_if_nested
      END PROGRAM test_target_if
