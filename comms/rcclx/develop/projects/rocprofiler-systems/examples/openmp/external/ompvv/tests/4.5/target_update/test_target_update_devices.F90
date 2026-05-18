!===--- test_target_update_devices.F90 --------------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
!
! This test checks if the target update directive works on different devices.
! We check two different variants.
! 1. setting up the default device with the API call omp_set_default_device()
! 2. using the device clause of the target update directive.
!
! Testing metodology uses an array that gets mapped into the device at first
! through target enter data. Then on each iteration we update the array in one
! device, create a compute region in that device, and then update it back
! We also record that the compute region is not executed in the host
! with the omp_is_initial_device() API call. Unfortunately 4.5 has no device
! number API call.
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM target_update_devices
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(test_set_default_dev() .ne. 0)
   OMPVV_TEST_VERBOSE(test_device() .ne. 0)
   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION test_set_default_dev()
      INTEGER :: num_dev, def_dev, summation, errors, i, dev
      LOGICAL, DIMENSION(0:5000) :: isHost ! Arbitrary number greater than the num
      INTEGER, DIMENSION(N) :: h_matrix
      CHARACTER(len = 400) :: numDevMsg, defDevMsg, initDevMsg, resultsMsg
      ! Initialize vars
      errors = 0
      summation = 0

      OMPVV_INFOMSG("test_set_default_dev()")

      ! Get number of devices
      num_dev = omp_get_num_devices()
      WRITE(numDevMsg, *) "num_devices:", num_dev
      OMPVV_INFOMSG(numDevMsg)

      def_dev = omp_get_default_device()
      WRITE(initDevMsg, *) "initial device is:", omp_get_initial_device()
      OMPVV_INFOMSG(initDevMsg)
      WRITE(defDevMsg, *) "default device is:", def_dev
      OMPVV_INFOMSG(defDevMsg)

      ! Mapping the array to all of the devices
      DO dev = 0, num_dev - 1
         CALL omp_set_default_device(dev)
         !$omp target enter data map (alloc: h_matrix)
      END DO

      ! Initialize the array
      DO i = 1, N
         h_matrix(i) = 0
      END DO

      ! Each device gets updated with the current array version,
      ! one gets added to each element in the array, and then
      ! the host gets the updated version
      DO dev = 0, num_dev - 1
         CALL omp_set_default_device(dev)
         !$omp target update to(h_matrix)
         !$omp target map(alloc: h_matrix) map(tofrom: isHost(dev))
            isHost(dev) = omp_is_initial_device()
         
            DO i = 1, N
               h_matrix(i) = h_matrix(i) + 1
            END DO
          !$omp end target
         !$omp target update from(h_matrix)
      END DO

      ! Unmap the matrix
      DO dev = 0, num_dev - 1
         CALL omp_set_default_device(dev)
         !$omp target exit data map(delete: h_matrix)
      END DO

      ! Checking results
      DO dev = 0, num_dev - 1
         IF (isHost(dev) .eqv. .TRUE.) THEN
            WRITE(resultsMsg, *) "device",dev,"ran on the host"
            OMPVV_INFOMSG(resultsMsg)
         ELSE
            WRITE(resultsMsg, *) "device",dev,"ran on the device"
            OMPVV_INFOMSG(resultsMsg)
         END IF
      END DO
      
      ! Checking results
      DO i = 1, N
         summation = summation + h_matrix(i)
      END DO
  
      OMPVV_TEST_AND_SET_VERBOSE(errors, (num_dev * N) .ne. summation)

      CALL omp_set_default_device(def_dev)
 
   test_set_default_dev = errors
   END FUNCTION test_set_default_dev

   INTEGER FUNCTION test_device() 
      INTEGER :: num_dev, def_dev, summation, errors, i, dev
      LOGICAL, DIMENSION(0:5000) :: isHost ! Arbitrary number greater than the num
      INTEGER, DIMENSION(N) :: h_matrix
      CHARACTER(len = 400) :: numDevMsg, defDevMsg, initDevMsg, resultsMsg
      ! Initialize vars
      errors = 0
      summation = 0

      OMPVV_INFOMSG("test_device()")
      
      ! Get number of devices
      num_dev = omp_get_num_devices()
      WRITE(numDevMsg, *) "num_devices:", num_dev
      OMPVV_INFOMSG(numDevMsg)

      def_dev = omp_get_default_device()
      WRITE(initDevMsg, *) "initial device is:", omp_get_initial_device()
      OMPVV_INFOMSG(initDevMsg)
      WRITE(defDevMsg, *) "default device is:", def_dev
      OMPVV_INFOMSG(defDevMsg)

      ! Mapping the array to all of the devices
      DO dev = 0, num_dev - 1
         !$omp target enter data map (alloc: h_matrix) device(dev)
      END DO

      ! Initialize the array
      DO i = 1, N
         h_matrix(i) = 0
      END DO

      ! Each device gets updated with the current array version,
      ! one gets added to each element in the array, and then
      ! the host gets the updated version
      DO dev = 0, num_dev - 1
         !$omp target update to(h_matrix) device(dev)
         !$omp target map(alloc: h_matrix) map(tofrom: isHost(dev)) device(dev)
            isHost(dev) = omp_is_initial_device()

            DO i = 1, N
               h_matrix(i) = h_matrix(i) + 1
            END DO
          !$omp end target
         !$omp target update from(h_matrix) device(dev)
      END DO
    
      ! Unmap the matrix
      DO dev = 0, num_dev - 1
         !$omp target exit data map(delete: h_matrix) device(dev)
      END DO

      ! Checking results
      DO dev = 0, num_dev - 1
         IF (isHost(dev) .eqv. .TRUE.) THEN
            WRITE(resultsMsg, *) "device",dev,"ran on the host"
            OMPVV_INFOMSG(resultsMsg)
         ELSE
            WRITE(resultsMsg, *) "device",dev,"ran on the device"
            OMPVV_INFOMSG(resultsMsg)
         END IF
      END DO

      ! Checking results
      DO i = 1, N
         summation = summation + h_matrix(i)
      END DO

      OMPVV_TEST_AND_SET_VERBOSE(errors, (num_dev * N) .ne. summation)

   test_device = errors
   END FUNCTION test_device
END PROGRAM target_update_devices
