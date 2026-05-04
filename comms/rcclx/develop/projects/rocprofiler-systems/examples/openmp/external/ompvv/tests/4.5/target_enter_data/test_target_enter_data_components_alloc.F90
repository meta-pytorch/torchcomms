!===--test_target_enter_data_components_alloc.F90 - derived data type map default-===!
! 
! OpenMP API Version 4.5 Nov 2015
!
! This test checks the mapping of derived data types (components) behavior.
! with the alloc modifier. Data should be copied over the device and
! from the device to the host
! 
!!===----------------------------------------------------------------------===!
#include "ompvv.F90"

#define N 100

      PROGRAM test_target_enter_data_components_alloc
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        INTEGER :: err_bf, err_af, i
        INTEGER, TARGET :: justATarget
        ! Defining a component for the test
        TYPE :: testingType 
          INTEGER, POINTER :: myPtr
          INTEGER :: myInt
          CHARACTER(len=100) :: myStr
          REAL:: myReal
          INTEGER, DIMENSION(N) :: myArr
        END TYPE testingType
        TYPE(testingType) :: myStruct
        TYPE(testingType) :: cpyStruct
        TYPE(testingType), dimension(10) :: myStructArr
        TYPE(testingType), dimension(10) :: cpyStructArr
        
        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_VERBOSE(test_map_derived_type_alloc() .ne. 0)

        OMPVV_REPORT_AND_RETURN()

        CONTAINS 
          ! Default mapping
          INTEGER FUNCTION test_map_derived_type_alloc()

            OMPVV_INFOMSG("Testing alloc mapping")

            OMPVV_GET_ERRORS(err_bf)

            ! Component mapping
            !$omp target enter data map(alloc: myStruct, myStructArr(:))

            ! Target region to confirm allocation
            !$omp target map(alloc: myStruct, myStructArr(:))
              myStruct%myInt = 5
              myStruct%myStr = "there"
              myStruct%myReal = 4.4
              myStruct%myArr(:) = 10
              myStruct%myPtr => justATarget
              DO i = 1, 10 
                myStructArr(i)%myInt = 5
                myStructArr(i)%myStr = "there"
                myStructArr(i)%myReal = 4.4
                myStructArr(i)%myArr(:) = 10
                myStructArr(i)%myPtr => justATarget
              END DO
            !$omp end target

            ! Target region to confirm that the allocation remained
            !$omp target map(alloc: myStruct, myStructArr(:)) &
            !$omp map(from: cpyStruct, cpyStructArr(:))
              cpyStruct%myInt = myStruct%myInt
              cpyStruct%myStr = myStruct%myStr
              cpyStruct%myReal = myStruct%myReal
              cpyStruct%myArr(:) = myStruct%myArr(:)

              DO i = 1, 10 
                cpyStructArr(i)%myInt = myStructArr(i)%myInt
                cpyStructArr(i)%myStr = myStructArr(i)%myStr
                cpyStructArr(i)%myReal = myStructArr(i)%myReal
                cpyStructArr(i)%myArr(1:N) = myStructArr(i)%myArr(1:N)
              END DO
            !$omp end target


            OMPVV_TEST_VERBOSE(cpyStruct%myInt /= 5)
            OMPVV_TEST_VERBOSE(cpyStruct%myStr .NE. 'there')
            OMPVV_TEST_VERBOSE(ABS(cpyStruct%myReal - 4.4) .GT. 0.0001)
            OMPVV_TEST_VERBOSE(ANY(cpyStruct%myArr /= 10))

            DO i = 1, 10 
              OMPVV_TEST_VERBOSE(cpyStructArr(i)%myInt /= 5)
              OMPVV_TEST_VERBOSE(cpyStructArr(i)%myStr .NE. 'there')
              OMPVV_TEST_VERBOSE(ABS(cpyStructArr(i)%myReal - 4.4) .GT. 0.0001)
              OMPVV_TEST_VERBOSE(ANY(cpyStructArr(i)%myArr /= 10))
            END DO

            ! This is not part of the test but necessary to avoid memory
            ! leaks
            !$omp target exit data map(delete: myStruct, myStructArr(:))

            OMPVV_GET_ERRORS(err_af)
            test_map_derived_type_alloc = err_af - err_bf

          END FUNCTION test_map_derived_type_alloc
      END PROGRAM test_target_enter_data_components_alloc

