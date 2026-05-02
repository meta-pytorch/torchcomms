!===--- test_task_detach.F90 -----------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This is a test of the 5.0 task detach clause. A task with the detach
! should not complete until its associated block and the event in the
! detach clause is fulfilled with the omp_fulfill_event clause. This test
! confirms that the task will not complete until these conditions are
! met by placing a dependent task after the detached task. Variables are
! set inside the task body and in the function that fulfills the event,
! and their values are checked after the depend to ensure they are set as
! expected. This test is based on the example of task detach presented
! by Michael Klemm at the 2018 OpenMPCon.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_task_detach
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  INTEGER :: final_errors = 0

  OMPVV_INFOMSG("test_task_detach")

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_AND_SET_VERBOSE(final_errors, task_detach() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

  CONTAINS
    INTEGER FUNCTION test_callback(event)
      INTEGER(kind=omp_event_handle_kind) :: event
      call omp_fulfill_event(event)
      test_callback = 1
      RETURN
    END FUNCTION test_callback

    INTEGER FUNCTION task_detach()
      INTEGER :: errors = 0
      INTEGER :: x = 0
      INTEGER :: y = 0
      INTEGER :: num_threads = -1
      INTEGER :: record_x = -1
      INTEGER :: record_y = -1
      INTEGER :: callback_result = 0
      INTEGER(kind=omp_event_handle_kind) :: flag_event
      CHARACTER(len=300) :: msgHelper

      !$omp parallel
      !$omp single

      !$omp task depend(out: y) detach(flag_event)
      y = y + 1
      !$omp end task

      !$omp task
      x = 1
      !$omp flush
      callback_result = test_callback(flag_event)
      !$omp end task

      !$omp task depend(inout: y)
      !$omp flush
      record_x = x
      record_y = y
      num_threads = omp_get_num_threads()
      !$omp end task

      !$omp end single
      !$omp end parallel

      OMPVV_ERROR_IF(num_threads .LT. 0, "Test ran with invalid number of teams")
      OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 0)
      
      OMPVV_ERROR_IF(callback_result .NE. 1, "Event-fulfilling callback did not return")
      OMPVV_TEST_AND_SET_VERBOSE(errors, callback_result .NE. 1)

      WRITE(msgHelper, *) "Test ran with one thread, &
           &so the results are not conclusive."
      OMPVV_WARNING_IF(num_threads .EQ. 1, msgHelper)

      OMPVV_TEST_AND_SET_VERBOSE(errors, record_x .NE. 1)

      WRITE(msgHelper, *) "Dependent task preceded event-fulfilling &
           &task, so detach did not work correctly."
      OMPVV_ERROR_IF(record_x .EQ. 0, msgHelper)

      WRITE(msgHelper, *) "Dependent task preceded detached task body, &
           &so depend did not work correctly."
      OMPVV_ERROR_IF(record_y .EQ. 0, msgHelper)

      WRITE(msgHelper, *) "Event-fulfilling task's recording variable &
           &was not set in the final task."
      OMPVV_ERROR_IF(record_x .EQ. -1, msgHelper)

      WRITE(msgHelper, *) "Detached task's recording variable was &
           &not set in the final task."
      OMPVV_ERROR_IF(record_y .EQ. -1, msgHelper)

      task_detach = errors
      RETURN
    END FUNCTION task_detach
  END PROGRAM test_task_detach
