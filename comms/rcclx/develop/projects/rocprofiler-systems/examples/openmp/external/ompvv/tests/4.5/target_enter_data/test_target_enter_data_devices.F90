!===---- test_target_enter_data_devices.F90 - using target data with multiple dev --===//
! 
! OpenMP API Version 4.5 Nov 2015
! 
! Using the device clause of the target enter data  to change the device to
! which the data is mapped
! 
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 5000
      PROGRAM test_target_enter_data_devices
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        LOGICAL :: isOffloading, isSharedEnv
        INTEGER :: num_dev
        CHARACTER(len=500) :: msgHelper

        OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
        OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)

        ! Reporting the number of devices
        num_dev = omp_get_num_devices()
        WRITE(msgHelper, '(A,I0)') "number of devices = ", num_dev
        OMPVV_INFOMSG(msgHelper)

        ! Warning if only one device
        WRITE(msgHelper, *) "The number of devices is 1, this test &
        &is inconclussive"
        OMPVV_WARNING_IF(num_dev == 1, msgHelper)

        !starting the tests
        OMPVV_TEST_VERBOSE(test_device_clause() .NE. 0)

        OMPVV_REPORT_AND_RETURN()

        CONTAINS 
          ! Testing set device with deviceclause
          INTEGER FUNCTION test_device_clause()
          INTEGER :: dev_data, dev_comp
          INTEGER, dimension(N) :: anArray, anotherArray

          OMPVV_INFOMSG("using device() clause")

          ! Iterate over all the devices and map the array with a diff
          ! value each device
          DO dev_data = 0, num_dev - 1
            ! Initialize the array
            anArray(:) = dev_data
            !$omp target enter data map(to: anArray(1:N)) &
            !$omp device(dev_data)
          END DO

          ! Iterate over all the devices and modify the data 
          ! that is already mapped
          DO dev_comp = 0, num_dev - 1
            !$omp target map(alloc: anArray(1:N)) &
            !$omp device(dev_comp)
              anArray(:) = anArray(:) + 1
            !$omp end target
          END DO ! dev_comp
              
          ! Iterate over all the devices and get the value back 
          ! without using target exit data (as that is for another
          ! tests 
          DO dev_data  = 0, num_dev - 1
            !$omp target map(from: anotherArray(1:N)) &
            !$omp device(dev_data)
              anotherArray(:) = anArray(:) - 1 
            !$omp end target
            OMPVV_TEST_VERBOSE(ANY(anotherArray /= dev_data))
            ! test for results
          END DO ! dev_comp

          ! This is not part of the test but for the sake of avoiding
          ! leaving allocated data on the device
          DO dev_data = 0, num_dev - 1
            !$omp target exit data map(delete: anArray(1:N)) &
            !$omp device(dev_data)
          END DO

          OMPVV_GET_ERRORS(test_device_clause)
          END FUNCTION test_device_clause
      END PROGRAM test_target_enter_data_devices

  
