!===---- test_target_teams_distribute_parallel_for.F90 - combined consutrct -===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! testing the combined construct target teams distribute parallel for
!
!===----------------------------------------------------------------------------------===//
#include "ompvv.F90"

#define ARRAY_SIZE 1024

      PROGRAM test_target_teams_distribute_parallel_for
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        
        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for() .NE. 0)

        OMPVV_REPORT_AND_RETURN()


        CONTAINS 
          INTEGER FUNCTION target_teams_distribute_parallel_for()
            INTEGER :: errors_bf, errors_af
            INTEGER :: i
            INTEGER, DIMENSION(ARRAY_SIZE) :: a = (/(1, i=0,ARRAY_SIZE - 1)/)
            INTEGER, DIMENSION(ARRAY_SIZE) :: b = (/(i, i=0,ARRAY_SIZE - 1)/)
            INTEGER, DIMENSION(ARRAY_SIZE) :: c = (/(2 * i, i=0,ARRAY_SIZE - 1)/)
            INTEGER :: num_teams = 0
            INTEGER, DIMENSION(ARRAY_SIZE) :: num_threads = (/(0, i=0,ARRAY_SIZE - 1)/)
            INTEGER :: alert_num_threads = 0
            CHARACTER(len=300) :: msgHelper
          
            OMPVV_INFOMSG("target_teams_distribute_parallel_for")
            OMPVV_GET_ERRORS(errors_bf)

            !$omp target teams distribute parallel do map(from:num_teams, num_threads) &
            !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE) num_threads(OMPVV_NUM_THREADS_DEVICE)
            DO i = 1, ARRAY_SIZE, 1
              !$omp atomic write 
              num_teams = omp_get_num_teams()
              num_threads(i) = omp_get_num_threads()
              a(i) = a(i) + b(i) * c(i);
            END DO

            DO i = 1, ARRAY_SIZE, 1
              OMPVV_TEST_AND_SET(errors_af, (a(i) .NE. (1 + (b(i) * c(i)))))
              IF (num_threads(i) .EQ. 1) THEN
                alert_num_threads = alert_num_threads + 1
              END IF 
            END DO
  
            ! Rise lack of parallelism alerts
            WRITE(msgHelper, *) "Test operated with one team.  &
            &Parallelism of teams distribute can't be guaranteed."
            OMPVV_WARNING_IF(num_teams == 1, msgHelper);

            WRITE(msgHelper, *) "Test operated with one thread &
            &in all the teams. Parallel clause had no effect" 
            OMPVV_WARNING_IF(alert_num_threads == ARRAY_SIZE, msgHelper);

            OMPVV_GET_ERRORS(errors_af)
            target_teams_distribute_parallel_for = errors_bf - errors_af
          END FUNCTION target_teams_distribute_parallel_for

      END PROGRAM test_target_teams_distribute_parallel_for

