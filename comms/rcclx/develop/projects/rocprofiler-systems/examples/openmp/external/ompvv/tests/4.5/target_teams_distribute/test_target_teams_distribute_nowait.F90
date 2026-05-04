!===--- test_target_teams_distribute_nowait.F90-----------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the nowait clause on a target teams distribute directive and
! uses a barrier to resyncronize the target regions.  Since we can't be sure
! that operations will be asyncronous, we can not test to make sure that
! the regions are executed asynchronously.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024
#define N_TASKS 16

PROGRAM test_target_teams_distribute_nowait
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  OMPVV_TEST_OFFLOADING
  errors = 0

  OMPVV_WARNING("This test does not throw an error if tasks fail to execute asynchronously, as this is still correct behavior. If execution is not asynchronous, we will throw a warning.")

  OMPVV_TEST_VERBOSE(test_nowait() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_nowait()
    USE ISO_Fortran_env, only: INT64
    INTEGER:: errors, x, y, z, was_async, my_ticket
    INTEGER,DIMENSION(N_TASKS):: order
    INTEGER(INT64),DIMENSION(N, N_TASKS):: work_storage
    INTEGER,DIMENSION(1):: ticket

    ticket(1) = 0
    errors = 0
    was_async = 0

    !$omp target enter data map(to: ticket(1:1), order(1:N_TASKS), my_ticket)

    DO x = 1, N_TASKS
       !$omp target teams distribute map(alloc: work_storage(1:N, x), ticket(1:1)) private(my_ticket) nowait
       DO y = 1, N
          work_storage(y, x) = 0
          DO z = 1, N*(N_TASKS - x)
             work_storage(y, x) = work_storage(y, x) + x*y*z
          END DO
          my_ticket = 0
          !$omp atomic capture
          ticket(1) = ticket(1) + 1
          my_ticket = ticket(1)
          !$omp end atomic
          order(x) = my_ticket
       END DO
       !$omp end target teams distribute
    END DO
    !$omp taskwait

    !$omp target exit data map(from:ticket(1:1), order(1:N_TASKS))

    IF (ticket(1) .ne. N_TASKS*N) THEN
       OMPVV_ERROR("The test registered a different number of target regions than were spawned")
       errors = 1
    END IF

    DO x = 2, N_TASKS
       IF (order(x) <= order(x - 1)) THEN
          was_async = 1
       END IF
    END DO

    OMPVV_WARNING_IF((was_async .eq. 0), "We could not detect asynchronous behavior between target regions")
    OMPVV_INFOMSG_IF((was_async .gt. 0), "Asynchronous behavior detected, this suggests nowait had an effect")

    test_nowait = errors
  END FUNCTION test_nowait
END PROGRAM test_target_teams_distribute_nowait
