!===--test_target_data_map_components_default.F90 - derived data type map default-===!
! 
! OpenMP API Version 4.5 Nov 2015
!
! This test checks the mapping of derived data types (components) behavior. 
! When using components, we want to make sure we can map them in
! different forms
! 
!!===----------------------------------------------------------------------===!
#include "ompvv.F90"

#define N 100

      PROGRAM test_target_data_map_components_default
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
        OMPVV_TEST_VERBOSE(test_map_derived_type_default() .ne. 0)

        OMPVV_REPORT_AND_RETURN()

        CONTAINS 
          ! Default mapping
          INTEGER FUNCTION test_map_derived_type_default()

            OMPVV_INFOMSG("Testing default mapping")

            OMPVV_GET_ERRORS(err_bf)

            ! Initializing component and array of component
            myStruct%myInt = 5
            myStruct%myStr = "there"
            myStruct%myReal = 4.4
            myStruct%myArr(:) = 10
            myStruct%myPtr => justATarget

            cpyStruct%myInt = 0
            cpyStruct%myStr = ""
            cpyStruct%myReal = 0.0
            cpyStruct%myArr(:) = 0
            cpyStruct%myPtr => justATarget
            
            DO i = 1, 10 
              myStructArr(i)%myInt = 5
              myStructArr(i)%myStr = "there"
              myStructArr(i)%myReal = 4.4
              myStructArr(i)%myArr(:) = 10
              myStructArr(i)%myPtr => justATarget

              cpyStructArr(i)%myInt = 0
              cpyStructArr(i)%myStr = ""
              cpyStructArr(i)%myReal = 0.0
              cpyStructArr(i)%myArr(:) = 0
              cpyStructArr(i)%myPtr => justATarget
            END DO

            !$omp target data map(myStruct, cpyStruct) &
            !$omp map(myStructArr(:), cpyStructArr(:))
              !$omp target map(alloc: myStruct, cpyStruct) &
              !$omp map(alloc:myStructArr(:), cpyStructArr(:))
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
            !$omp end target data


            OMPVV_TEST_VERBOSE(cpyStruct%myInt /= 5)
            OMPVV_TEST_VERBOSE(cpyStruct%myStr .NE. 'there')
            OMPVV_TEST_VERBOSE(ABS(cpyStruct%myReal - 4.4) .GT. 0.0001)
            OMPVV_TEST_VERBOSE(ANY(cpyStruct%myArr /= 10))
            OMPVV_TEST_VERBOSE(.NOT. ASSOCIATED(myStruct%myPtr, justATarget))
            OMPVV_TEST_VERBOSE(.NOT. ASSOCIATED(cpyStruct%myPtr, justATarget))

            DO i = 1, 10 
              OMPVV_TEST_VERBOSE(cpyStructArr(i)%myInt /= 5)
              OMPVV_TEST_VERBOSE(cpyStructArr(i)%myStr .NE. 'there')
              OMPVV_TEST_VERBOSE(ABS(cpyStructArr(i)%myReal - 4.4) .GT. 0.0001)
              OMPVV_TEST_VERBOSE(ANY(cpyStructArr(i)%myArr /= 10))
              OMPVV_TEST_VERBOSE(.NOT. ASSOCIATED(myStructArr(i)%myPtr, justATarget))
              OMPVV_TEST_VERBOSE(.NOT. ASSOCIATED(cpyStructArr(i)%myPtr, justATarget))
            END DO

            OMPVV_GET_ERRORS(err_af)
            test_map_derived_type_default = err_af - err_bf

          END FUNCTION test_map_derived_type_default
      END PROGRAM test_target_data_map_components_default

