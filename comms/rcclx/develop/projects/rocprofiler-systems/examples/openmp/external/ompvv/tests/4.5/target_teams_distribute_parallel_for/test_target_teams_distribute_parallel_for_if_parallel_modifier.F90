!===--- test_target_teams_distribute_parallel_for_if_no_modifier.F90 ---------===!
!
! OpenMP API Version 4.5 Nov 2015
!
! In this test we want to try to check if the if clause is working 
! when used with the combined construct target teams distribute
! parallel for and the parallel modifier is specified.
! To do this we check if offloading is working, if it is not, it won't
! be possible for us to tell if the test passed or not, since it depends
! on offloading capabilities.
!
! The if has a directive-name-modifier that specifies to which directive
! the if applies to (either both directives, to the target or to the
! parallel). 
! We create three tests, one for no directive, another one for the
! target 
! directive and another one for the parallel directive.
!
!===--------------------------------------------------------------------------===!

#include "ompvv.F90"

#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_if_parallel_modifier
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(target_teams_distribute_if_parallel_modifier() .ne. 0)
   OMPVV_REPORT_AND_RETURN()  

CONTAINS 
   LOGICAL FUNCTION checkPreconditions()
      LOGICAL :: isOffloading
      INTEGER :: i, raiseWarningDevice, raiseWarningHost
      INTEGER, DIMENSION(N) :: init_num_threads_dev, init_num_threads_host

      OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
 
      ! Initialize init_num_threads_dev and init_num_threads_host arrays
      DO i = 1, N
         init_num_threads_dev(i) = 0
         init_num_threads_host(i) = 0
      END DO

      ! Get the init_num_threads for host and device, if it is 1 then parallel
      ! cannot be tested
      !$omp target teams distribute parallel do num_threads(OMPVV_NUM_THREADS_DEVICE)
      DO i = 1, N
         IF (omp_get_thread_num() .eq. 0) THEN
            init_num_threads_dev(i) = omp_get_num_threads()
         END IF
      END DO

      !$omp parallel do num_threads(OMPVV_NUM_THREADS_DEVICE)
      DO i = 1, N
         IF (omp_get_thread_num() .eq. 0) THEN
            init_num_threads_host(i) = omp_get_num_threads()
         END IF
      END DO

      raiseWarningDevice = 0
      raiseWarningHost = 0

      DO i = 1, N
         IF (init_num_threads_dev(i) .eq. 1) THEN
            raiseWarningDevice = raiseWarningDevice + 1
         END IF
         IF (init_num_threads_host(i) .eq. 1) THEN
            raiseWarningHost = raiseWarningHost + 1
         END IF
      END DO

      OMPVV_WARNING_IF(raiseWarningDevice .ne. 0, "Initial number of threads on device was 1. It is not possible to test the if for the parallel directive")
 
      OMPVV_WARNING_IF(raiseWarningHost .ne. 0, "Initial number of threads on host was 1. It is not possible to test the if for the parallel directive")

   checkPreconditions = isOffloading
   END FUNCTION checkPreconditions

   INTEGER FUNCTION target_teams_distribute_if_parallel_modifier()
      INTEGER, DIMENSION(N) :: a, warning
      INTEGER :: attempt, errors, i, raiseWarning 
      LOGICAL :: testOffload

      testOffload = checkPreconditions() 

      attempt = 0
      errors = 0 
 

      DO i = 1, N
         a(i) = 0
         warning(i) = 0
      END DO

      DO attempt = 1, NUM_ATTEMPTS
         !$omp target teams distribute parallel do if(parallel: attempt .gt. ATTEMPT_THRESHOLD) &
         !$omp& map(tofrom: a, warning) num_threads(OMPVV_NUM_THREADS_DEVICE)
         DO i = 1, N
            IF (omp_is_initial_device() .eqv. .TRUE.) THEN
               a(i) = a(i) + 10 ! This +10 should not happen
            END IF
            
            IF (attempt .gt. ATTEMPT_THRESHOLD) THEN
               IF (omp_get_num_threads() .eq. 1) THEN
                  warning(i) = warning(i) + 1
               END IF
            ELSE
               IF (omp_get_num_threads() .ne. 1) THEN
                  a(i) = a(i) + 10
               ELSE
                  a(i) = a(i) + 1
               END IF
            END IF
         END DO
      END DO

      raiseWarning = 0
      DO i = 1, N
         OMPVV_TEST_AND_SET(errors, a(i) .ne. ATTEMPT_THRESHOLD)
         IF (warning(i) .ne. 0) THEN
            raiseWarning = raiseWarning + 1
         END IF
      END DO
      
      OMPVV_WARNING_IF(raiseWarning .eq. (N * (NUM_ATTEMPTS - ATTEMPT_THRESHOLD)), "The number of threads was 1 when a number > 1 was expected. if(parallel:true). Not a compliance error in the specs")

      OMPVV_ERROR_IF(errors .ne. 0, "error in if(parallel: modifier). Possible causes are: the execution occurred in the host even though it should not affect the target region. The number of threads was > 1 when if(false).")
 
      OMPVV_WARNING_IF(testOffload .neqv. .TRUE.," Offloading is not enabled for this test, behavior of this test cannot be verified without offloading enabled.")
   
   target_teams_distribute_if_parallel_modifier = errors
   END FUNCTION target_teams_distribute_if_parallel_modifier   
   
END PROGRAM test_target_teams_distribute_parallel_for_if_parallel_modifier
