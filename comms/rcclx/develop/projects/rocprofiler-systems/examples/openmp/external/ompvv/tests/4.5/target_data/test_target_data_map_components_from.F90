!===--test_target_data_map_components_from.F90 - derived data type map to -===!
!
! OpenMP API Version 4.5 Nov 2015
!
! This test check for the to mapping of components in both regular
! variables and arrays
!
!!===----------------------------------------------------------------------===!
#include "ompvv.F90"

#define N 100

      PROGRAM test_target_data_map_components_from
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        INTEGER :: err_bf, err_af, i
        INTEGER, TARGET :: justATarget
        CHARACTER(len=400) :: auxMessage
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

        OMPVV_TEST_VERBOSE(test_map_derived_type_from() .ne. 0)

        OMPVV_REPORT_AND_RETURN()

        CONTAINS
          ! Default mapping
          INTEGER FUNCTION test_map_derived_type_from()

            OMPVV_INFOMSG("Testing from mapping")

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

            !$omp target data map(from: myStruct) &
            !$omp map(from: myStructArr(:))
              !$omp target map(alloc: myStruct, myStructArr) &
              !$omp map(tofrom: cpyStruct, cpyStructArr(:))
                ! Obtain the original value of myStruct
                ! to check if it was copied to as well
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

                ! Modify myStruct and array to get the value from the
                ! host
                myStruct%myInt = 6
                myStruct%myStr = "you"
                myStruct%myReal = 6.4
                myStruct%myArr(:) = 100

                DO i = 1, 10
                  myStructArr(i)%myInt = 6
                  myStructArr(i)%myStr = "you"
                  myStructArr(i)%myReal = 6.4
                  myStructArr(i)%myArr(:) = 100
                END DO
              !$omp end target
            !$omp end target data


            ! Checking that the data was copied to the device
            WRITE(auxMessage, *) "Array seemed to have been copied to &
                 & the device when using the from modifier."
            OMPVV_WARNING_IF(cpyStruct%myInt == 5, auxMessage)
            OMPVV_WARNING_IF(cpyStruct%myStr .EQ. 'there', auxMessage)
            OMPVV_WARNING_IF(ABS(cpyStruct%myReal - 4.4) .LT. 0.0001, auxMessage)
            OMPVV_WARNING_IF(ALL(cpyStruct%myArr == 10), auxMessage)
            OMPVV_WARNING_IF(.NOT. ASSOCIATED(cpyStruct%myPtr, justATarget), auxMessage)

            DO i = 1, 10
               OMPVV_WARNING_IF(cpyStructArr(i)%myInt == 5, auxMessage)
               OMPVV_WARNING_IF(cpyStructArr(i)%myStr .EQ. 'there', auxMessage)
               OMPVV_WARNING_IF(ABS(cpyStructArr(i)%myReal - 4.4) .LT. 0.0001, auxMessage)
               OMPVV_WARNING_IF(ANY(cpyStructArr(i)%myArr == 10), auxMessage)
               OMPVV_WARNING_IF(.NOT. ASSOCIATED(cpyStructArr(i)%myPtr, justATarget), auxMessage)
            END DO

            ! Checking that the data was copied from the device
            OMPVV_TEST_VERBOSE(myStruct%myInt /= 6)
            OMPVV_TEST_VERBOSE(myStruct%myStr .NE. 'you')
            OMPVV_TEST_VERBOSE(ABS(myStruct%myReal - 6.4) .GT. 0.0001)
            OMPVV_TEST_VERBOSE(ANY(myStruct%myArr /= 100))
            OMPVV_TEST_VERBOSE(ASSOCIATED(myStruct%myPtr, justATarget))

            DO i = 1, 10
              OMPVV_TEST_VERBOSE(myStructArr(i)%myInt /= 6)
              OMPVV_TEST_VERBOSE(myStructArr(i)%myStr .NE. 'you')
              OMPVV_TEST_VERBOSE(ABS(myStructArr(i)%myReal - 6.4) .GT. 0.0001)
              OMPVV_TEST_VERBOSE(ANY(myStructArr(i)%myArr /= 100))
              OMPVV_TEST_VERBOSE(ASSOCIATED(myStructArr(i)%myPtr, justATarget))
            END DO

            OMPVV_GET_ERRORS(err_af)
            test_map_derived_type_from = err_af - err_bf

          END FUNCTION test_map_derived_type_from
      END PROGRAM test_target_data_map_components_from
