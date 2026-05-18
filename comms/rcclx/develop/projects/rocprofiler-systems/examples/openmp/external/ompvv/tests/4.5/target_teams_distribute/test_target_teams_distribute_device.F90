!===--- test_target_teams_distribute_device.F90------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the device clause to indicate which device should execute the
! given target regions.  The test uses the separate device data environments to
! ensure that operations are executed on the specified device.  If only one device
! is available, the test issues a warning.
!
! By having a separate initialization of the same array on each device at the
! same time, if all operations were occuring on the same device, we would expect
! the same results from each device and it wouldn't be able to give proper answers
! for each initialization.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  errors = 0
  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_VERBOSE(test_multiple_devices() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_multiple_devices()
    INTEGER :: x, dev, num_devices, total_errors
    INTEGER, DIMENSION(N) :: a, b
    INTEGER, ALLOCATABLE :: c(:, :)
    INTEGER, ALLOCATABLE :: num_teams(:), errors(:)
    CHARACTER(len=100) :: message

    num_devices = omp_get_num_devices()
    total_errors = 0

    ALLOCATE(num_teams(num_devices))
    ALLOCATE(errors(num_devices))
    ALLOCATE(c(N, num_devices))

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x, :) = 0
    END DO

    DO x = 1, num_devices
       num_teams(x) = 0
       errors(x) = 0
    END DO

    DO dev = 1, num_devices
       !$omp target enter data map(to: a(1:N), b(1:N), c(1:N, dev:dev), &
       !$omp& num_teams(dev:dev)) device(dev - 1)
    END DO

    DO dev = 1, num_devices
       !$omp target teams distribute map(alloc: a(1:N), b(1:N), &
       !$omp& c(1:N, dev:dev), num_teams(dev:dev)) device(dev - 1)
       DO x = 1, N
          IF (omp_get_team_num() .eq. 0) THEN
             num_teams(dev) = omp_get_num_teams()
          END IF
          c(x, dev) = a(x) + b(x) + dev
       END DO
    END DO

    DO dev = 1, num_devices
       !$omp target exit data map(from: c(1:N, dev:dev), &
       !$omp& num_teams(dev:dev)) map(delete: a(1:N), b(1:N)) device(dev - 1)
       DO x = 1, N
          IF (c(x, dev) .ne. (1 + dev + x)) THEN
             errors(dev) = errors(dev) + 1
          END IF
       END DO
       total_errors = total_errors + errors(dev)
    END DO

    DO dev = 1, num_devices
       IF (errors(dev) .eq. 0 .and. num_teams(dev) .eq. 1) THEN
          WRITE(message, '(A,I0)') "Test operated with one team on &
               &device: ", dev
          OMPVV_INFOMSG(message)
       ELSE IF (errors(dev) .ne. 0) THEN
          WRITE(message, '(A,I0)') "Test failed on device: ", dev
          OMPVV_INFOMSG(message)
       END IF
    END DO
    test_multiple_devices = total_errors
  END FUNCTION test_multiple_devices
END PROGRAM test_target_teams_distribute_device
