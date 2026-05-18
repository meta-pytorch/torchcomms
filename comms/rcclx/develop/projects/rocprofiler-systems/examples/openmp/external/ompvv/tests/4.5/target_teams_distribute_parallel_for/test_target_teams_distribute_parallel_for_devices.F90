!===--- test_target_teams_distribute_parallel_for_devices.F90 ---------------===//
!
! OpenMP API Version 4.5 Nov 2015
! 
! Testing for multiple devices checking if it is possible to send work and data 
! to different devices with the device clause used with omp target teams distribute 
! parallel for
! 
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_devices
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_OFFLOADING

   OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_devices() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_teams_distribute_parallel_for_devices()
      INTEGER :: num_dev, errors, i, dev, dev2, initial_dev
      INTEGER, DIMENSION(N) :: a
      LOGICAL, DIMENSION(0:N) :: isHost
      CHARACTER(len=400) :: numDeviceMsg, hostOrDevMsg

      errors = 0

      initial_dev = omp_get_initial_device()
      num_dev = omp_get_num_devices()
      OMPVV_WARNING_IF(num_dev .le. 1, "Testing devices clause without multiple devices")

      WRITE(numDeviceMsg, *) "Number of devices =", num_dev
      OMPVV_INFOMSG(numDeviceMsg)

      DO i = 1, N
         a(i) = 1
      END DO

      DO dev = 0, num_dev
         ! Include the initial device in the tests - but honor that before
         ! OpenMP < 5.1, num_dev might not be a valid device number
         IF (i == num_dev) THEN
           dev2 = initial_dev
         ELSE
           dev2 = dev
         END IF
         ! This testcase assumes that 'map' actually places 'a' into device memory
         ! and is not a no op - not even with (unified) shared memory.
         !$omp target enter data map(to: a) device(dev2)
      END DO

      DO dev = 0, num_dev
         IF (dev == num_dev) THEN
           dev2 = initial_dev
         ELSE
           dev2 = dev
         END IF
         ! The implicit 'map(tofrom: a)' does not imply an update of 'a' as 'a'
         ! is already present on the device.
         !$omp target teams distribute parallel do device(dev2) map(tofrom: isHost)
            DO i = 1, N
               IF ((omp_get_team_num() .eq. 0) .and. (omp_get_thread_num() .eq. 0)) THEN
                  isHost(dev) = omp_is_initial_device()
               END IF
               a(i) = a(i) + dev
            END DO
      END DO

      ! Check the host value (dev == num_dev) first in order to avoid that it
      ! gets overridden by the device-memory versions (dev = 0 ... num_dev - 1)
      DO dev = num_dev, 0, -1
         IF (dev == num_dev) THEN
           dev2 = initial_dev
         ELSE
           dev2 = dev
         END IF
         !$omp target exit data map(from: a) device(dev2)
         IF (isHost(dev) .eqv. .true.) THEN 
            WRITE(hostOrDevMsg, *) "Device", dev, "ran on the host"
         END IF
         IF (isHost(dev) .eqv. .false.) THEN
            WRITE(hostOrDevMsg, *) "Device", dev, "ran on the device"
         END IF
         OMPVV_INFOMSG(hostOrDevMsg)
         DO i = 1, N
            OMPVV_TEST_AND_SET(errors, a(i) .ne. (1 + dev))
         END DO
      END DO
       

      target_teams_distribute_parallel_for_devices = errors
   END FUNCTION target_teams_distribute_parallel_for_devices
END PROGRAM test_target_teams_distribute_parallel_for_devices

