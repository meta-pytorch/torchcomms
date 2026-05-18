!===------ ompvv.F ------------- OMPVV HEADER FILE FORTRAN -----===//
!
! Header file for OMP Validation and verification test suite
! TODO: Most of the macros are not used. It was not found a way to
! make the compilers accept them as they were developed for the
! ompvv.F90 from which this filed was copied and modified from
!
!===--------------------------------------------------------------===//
#define __LINENUMBER__ __LINE__
!#define __FILENAME__ __FILE__(INDEX(__FILE__,"/", back=.true.)+1:)
#define __FILENAME__ __FILE__
#define OMPVV_HEADER_FMT(header) '("[header ",A,":",I0,"] ",A)'
#define OMPVV_HEADER_RESULT_FMT '("[OMPVV_RESULT ",A,"] Test ",A," on the ",A,".")'

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif

#ifndef OMPVV_MODULE_REQUIRES_LINE
#define OMPVV_MODULE_REQUIRES_LINE
#endif

! Macro for output of information, warning and error messages
#ifdef VERBOSE_MODE
#define OMPVV_WARNING_HELPER(message, filename, line) WRITE(*, OMPVV_HEADER_FMT(OMPVV_WARNING)) TRIM(clean_fn(filename)), line, TRIM(message)
#define OMPVV_WARNING(message) OMPVV_WARNING_HELPER(message,__FILENAME__,__LINE__)
#define OMPVV_WARNING_IF(condition, message) IF (condition) OMPVV_WARNING(message)
#define OMPVV_ERROR_HELPER(msg, file, line)  WRITE(ERROR_UNIT, OMPVV_HEADER_FMT(OMPVV_ERROR)) TRIM(clean_fn(file)), line, TRIM(msg)
#define OMPVV_ERROR(message) OMPVV_ERROR_HELPER(message,__FILENAME__,__LINE__)
#define OMPVV_ERROR_IF(condition, message) IF (condition) OMPVV_ERROR(message)
#define OMPVV_INFOMSG_HELPER(message, filename, line) WRITE(*, OMPVV_HEADER_FMT(OMPVV_INFO)) TRIM(clean_fn(filename)), line, TRIM(message)
#define OMPVV_INFOMSG(message) OMPVV_INFOMSG_HELPER(message, __FILENAME__, __LINE__)
#define OMPVV_INFOMSG_IF(condition, message) IF (condition) OMPVV_INFOMSG(message)
! END IF VERBOSE_MODE
#endif

! NONE VERBOSE MODE
#ifndef VERBOSE_MODE

#define OMPVV_WARNING_HELPER(message, filename, line) CONTINUE ! message, filename, line
#define OMPVV_WARNING(message) CONTINUE !message
#define OMPVV_WARNING_IF(condition, message) CONTINUE !message
#define OMPVV_ERROR_HELPER(message, filename, line) CONTINUE ! message, filename, line
#define OMPVV_ERROR(message) CONTINUE !message
#define OMPVV_ERROR_IF(condition, message) CONTINUE !message
#define OMPVV_INFOMSG_HELPER(message, filename, line) CONTINUE ! message, filename, line
#define OMPVV_INFOMSG(message) CONTINUE !message
#define OMPVV_INFOMSG_IF(condition, message) CONTINUE !message

! END IF VERBOSE_MODE
#endif

! Macros to provide thread and team nums if they are not specified
#ifndef OMPVV_NUM_THREADS_DEVICE
#define OMPVV_NUM_THREADS_DEVICE 8
#endif

#ifndef OMPVV_NUM_TEAMS_DEVICE
#define OMPVV_NUM_TEAMS_DEVICE 8
#endif

#ifndef OMPVV_NUM_THREADS_HOST
#define OMPVV_NUM_THREADS_HOST 8
#endif

! Macro for checking if offloading is enabled or not
#define OMPVV_TEST_OFFLOADING call test_offloading(__FILENAME__, __LINE__)

! Macro for checking if offloading is enabled or not and set a variable
! with the result
#define OMPVV_TEST_AND_SET_OFFLOADING(var2set) var2set = test_and_set_offloading(__FILENAME__, __LINE__)

! Macro for testing for errors
#define OMPVV_TEST(condition) call test_error(condition, __FILENAME__, __LINE__)

! Macro for testing for errors
#define OMPVV_TEST_VERBOSE(condition) call test_error_verbose(condition,"condition", __FILENAME__, __LINE__)

! Macro for setting errors on condition
#define OMPVV_TEST_AND_SET(err, condition) err = err + test_and_set(condition, __FILENAME__, __LINE__)

! Macro for setting errors on condition and displaying an error if 
! something went wrong
#define OMPVV_TEST_AND_SET_VERBOSE(err, condition) err = err + test_and_set_verbose(condition, "condition", __FILENAME__, __LINE__)

! Macro for reporting results
#define OMPVV_REPORT() call report_errors(__FILENAME__)

! Macro fo getting and setting the current number of errors
#define OMPVV_GET_ERRORS(err) err = get_errors()
#define OMPVV_SET_ERRORS(err) call set_errors(err)

! Macro for correct exit code
#define OMPVV_RETURN(err) if( err .ne. 0) stop EXIT_FAILURE

#define OMPVV_REPORT_AND_RETURN() if( report_and_set_errors(__FILENAME__) .ne. 0) stop EXIT_FAILURE

! Macro to report warning if it is a shared environment
#define OMPVV_TEST_SHARED_ENVIRONMENT call test_shared_environment(__FILENAME__, __LINE__)

! Macro to report warning if it is a shared environment and set a 
! variable for further use
#define OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(var2set) var2set = test_and_set_shared_environment(__FILENAME__, __LINE__)

! Auxiliar module used for standarized output and testing
      module ompvv_lib
        use omp_lib
        use iso_fortran_env
        implicit none
       OMPVV_MODULE_REQUIRES_LINE
       LOGICAL, PRIVATE :: ompvv_isHost = .TRUE.
       INTEGER, PRIVATE :: ompvv_errors = 0
       LOGICAL, PRIVATE :: ompvv_sharedEnv
      contains
      function clean_fn(fn)
        CHARACTER(len = *) :: fn
        CHARACTER(len = 400) :: clean_fn
        INTEGER :: fn_cut_point

        ! Avoid unused variables warning
        fn_cut_point = SCAN(fn, "/", .TRUE.)
        IF (fn_cut_point .GT. 0) THEN
          clean_fn = TRIM(fn(fn_cut_point + 1:))
        ELSE
          clean_fn = fn
        END IF
      end function

! Sets the isHost variable checking if it is device or hosts
      subroutine test_offloading_probe(ompvv_isHost)
        LOGICAL ompvv_isHOST

        ompvv_isHost = .FALSE.
!$omp target map(from:ompvv_isHost)
        ompvv_isHost = omp_is_initial_device()
!$omp end target
      end subroutine test_offloading_probe

      ! test offloading prints if offloading is enabled or not
      subroutine test_offloading(fn ,ln)
        CHARACTER(len=*) fn
        CHARACTER(len=500) clean
        INTEGER ln

      ! Avoid unused variables warning
        clean = TRIM(clean_fn(fn))

        IF (.FALSE.) THEN
          ln = ln
        END IF

        call test_offloading_probe(ompvv_isHost)
        IF (ompvv_isHost) THEN
          !OMPVV_INFOMSG_HELPER("Test is running on host",clean ,ln)
          WRITE(*, OMPVV_HEADER_FMT(OMPVV_INFO)) 
     &     TRIM(clean_fn(__FILE__)), __LINE__, 
     &     TRIM("Test is running on host")
        ELSE
          !OMPVV_INFOMSG_HELPER("Test is running on device",clean ,ln)
           WRITE(*, OMPVV_HEADER_FMT(OMPVV_INFO))
     &     TRIM(clean_fn(__FILE__)), __LINE__,
     &     TRIM("Test is running on device")
        END IF
      end subroutine test_offloading

      ! This function check if offloading is enabled and return true or false
      function test_and_set_offloading(fn , ln)
        CHARACTER(len=*) fn
        INTEGER ln
        LOGICAL test_and_set_offloading

        call test_offloading(fn, ln)
        test_and_set_offloading = .NOT. ompvv_isHost

      end function test_and_set_offloading

      ! Function to test an error and register in the error variable
      subroutine test_error(condition, fn, ln)
        LOGICAL condition
        CHARACTER(len=*) fn
        INTEGER ln

        ! Avoid unused variables warning
        IF (.FALSE.) THEN
          fn = fn
          ln = ln
        END IF

        IF (condition) ompvv_errors = ompvv_errors + 1
      end subroutine test_error

      ! Function to test an error condition and report it to the user
      subroutine test_error_verbose(condition, conditionStr, fn, ln)
        LOGICAL condition
        CHARACTER(len=*) conditionStr
        CHARACTER(len=*) fn
        CHARACTER(len=500) clean, clean_conditionStr
        INTEGER ln, condition_clean_pos

        clean = TRIM(clean_fn(fn))
        clean_conditionStr = conditionStr
        ! Avoid unused variables warning
        IF (.FALSE.) THEN
          ln = ln
        END IF
        ! cleaning condition % causes to fail. replace with '.'
        condition_clean_pos = SCAN(clean_conditionStr, "%")
        DO WHILE (condition_clean_pos /= 0)
          clean_conditionStr (condition_clean_pos : condition_clean_pos 
     &    + 1) = "."
          condition_clean_pos = SCAN(clean_conditionStr , "%")
        END DO
        clean_conditionStr = " Condition " //TRIM(clean_conditionStr) //
     &  " failed "

        IF (condition) THEN
          ompvv_errors = ompvv_errors + 1
          !OMPVV_ERROR_HELPER(clean_conditionStr, clean, ln)
          WRITE(ERROR_UNIT, OMPVV_HEADER_FMT(clean_conditionStr))
     &    __FILE__, __LINE__, " Condition failed "
          !OMPVV_ERROR_HELPER(" Condition failed ", fn, ln)
        END IF
      end subroutine test_error_verbose

      ! Function to test an error condition and return the current value of ompvv_errors
      function test_and_set(condition, fn, ln)
        LOGICAL condition
        CHARACTER(len=*) fn
        INTEGER ln
        INTEGER test_and_set, err_bf
        err_bf = ompvv_errors

        call test_error(condition, fn, ln)

        test_and_set = ompvv_errors - err_bf
      end function test_and_set

      ! Function to test an error condition, return the current value of ompvv_errors
      ! and report the current value to the user STDIO
      function test_and_set_verbose(condition, conditionStr, fn, ln)
        LOGICAL condition
        CHARACTER(len=*) conditionStr
        CHARACTER(len=*) fn
        INTEGER ln
        INTEGER test_and_set_verbose, err_bf
        err_bf = ompvv_errors

        call test_error_verbose(condition, conditionStr, fn, ln)

        test_and_set_verbose = ompvv_errors - err_bf
      end function test_and_set_verbose

      ! Function to report ompvv_errors
      subroutine report_errors(fn)
        CHARACTER(len=*) fn
        CHARACTER(len=50) message2dis
        CHARACTER(len=500) clean

        clean = TRIM(clean_fn(fn))

        WRITE(message2dis, '(A,I0)') "The value of errors is ", 
     &     ompvv_errors
        message2dis = TRIM(message2dis)

        IF (ompvv_errors /= 0) THEN
          WRITE(*, OMPVV_HEADER_FMT(OMPVV_INFOMSG)) 
     &        TRIM(clean_fn(__FILE__)), __LINE__, TRIM(message2dis)
          !OMPVV_INFOMSG(message2dis)
          IF (ompvv_isHost) THEN
            WRITE(*,OMPVV_HEADER_RESULT_FMT)
     &      TRIM(clean), "failed",
     &      "host"
          ELSE
            WRITE(*,OMPVV_HEADER_RESULT_FMT)
     &      TRIM(clean), "failed",
     &      "device"
          END IF
        ELSE
          WRITE(*, OMPVV_HEADER_FMT(OMPVV_INFOMSG))
     &        TRIM(clean_fn(__FILE__)), __LINE__, 
     &        TRIM("Test ran with no errors")
          !OMPVV_INFOMSG("Test ran with no errors")
          IF (ompvv_isHost) THEN
            WRITE(*,OMPVV_HEADER_RESULT_FMT)
     &      TRIM(clean), "passed",
     &      "host"
          ELSE
            WRITE(*,OMPVV_HEADER_RESULT_FMT)
     &      TRIM(clean), "passed",
     &      "device"
          END IF
        END IF

      end subroutine report_errors

      function report_and_set_errors(fn)
        CHARACTER(len=*) fn
        INTEGER report_and_set_errors

        call report_errors(fn)

        report_and_set_errors = ompvv_errors
      end function report_and_set_errors

      function get_errors()
        INTEGER get_errors

        get_errors = ompvv_errors
      end function get_errors

      subroutine set_errors(err)
        INTEGER err

        ompvv_errors = err
      end subroutine set_errors

      ! Macro to check if it is a shared data environment
      subroutine test_shared_environment_probe(ompvv_isHost)
        INTEGER isSharedProb
        LOGICAL ompvv_isHost

        isSharedProb = 0
        ompvv_sharedEnv = .FALSE.

        ! checking for isHost
        call test_offloading_probe(ompvv_isHost)

!$omp target map(to: isSharedProb)
        isSharedProb = 1
!$omp end target

        IF ((ompvv_isHost .EQV. .FALSE. ) .AND. (isSharedProb .EQ. 1))
     &  THEN
          ompvv_sharedEnv = .TRUE.
        END IF
      end subroutine test_shared_environment_probe

      subroutine test_shared_environment(fn, ln)
        CHARACTER(len=*) fn
        CHARACTER(len=500) clean
        INTEGER ln
#ifdef VERBOSE_MODE
        CHARACTER(len=*) msg
        PARAMETER (msg = "This tests is running on " //
     &   "a shared data environment between host and device. This " //
     &   "may cause errors")
#endif

        clean = TRIM(clean_fn(fn))
        ! Avoid unused variables warning
        IF (.FALSE.) THEN
          ln = ln
        END IF

        call test_shared_environment_probe(ompvv_sharedEnv)

#ifdef VERBOSE_MODE
        IF (ompvv_sharedEnv) THEN
          !OMPVV_WARNING_HELPER(msg, clean, ln)
          WRITE(*, OMPVV_HEADER_FMT(OMPVV_WARNING)) 
     &     TRIM(clean_fn(__FILENAME__)), __LINE__, TRIM(msg)
        END IF
#endif
      end subroutine test_shared_environment

      function test_and_set_shared_environment(fn, ln)
        CHARACTER(len=*) fn
        INTEGER ln
        LOGICAL :: test_and_set_shared_environment

        call test_shared_environment(fn, ln)

        test_and_set_shared_environment = ompvv_sharedEnv

      end function test_and_set_shared_environment


      end module ompvv_lib
