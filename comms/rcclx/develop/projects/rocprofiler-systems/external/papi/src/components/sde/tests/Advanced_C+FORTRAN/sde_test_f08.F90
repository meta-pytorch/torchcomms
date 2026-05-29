        program test_F_interface
          use, intrinsic :: ISO_C_BINDING
          use  :: ISO_FORTRAN_ENV
          use  :: papi_sde_fortran_wrappers

          implicit none

          TYPE(C_ptr) :: handle
          TYPE(C_ptr) :: quantile
          integer, pointer :: quantile_f
          integer(kind=C_LONG_LONG) :: values(22), values_to_write(1)

          integer(kind=C_LONG_LONG), target :: ev_cnt1, ev_cnt2
          integer(kind=C_INT), target :: ev_cnt3_int
          real, target :: ev_cnt3_float
          real (KIND=KIND(0.0)),   target :: ev_cnt4_float
          real (KIND=KIND(0.0D0)) :: value_d
          integer :: i, ret_val, error
          integer :: eventset, eventset2, eventcode, junk, codes(3)

          real, target :: internal_variable
          integer :: internal_variable_int
          integer :: all_tests_passed

          character(:), allocatable :: arg
          integer :: arglen, stat
          integer :: be_verbose

          interface
            function callback_t(param) result(ret_val)
              use, intrinsic :: ISO_C_BINDING, only: C_LONG_LONG
              real :: param
              integer(kind=C_LONG_LONG) :: ret_val
            end function callback_t
          end interface

          interface
            function rounding_error(param) result(ret_val)
              real (KIND=KIND(0.0D0)) :: param, ret_val
            end function rounding_error
          end interface

          procedure(callback_t) :: f08_callback

          all_tests_passed = 1

          ev_cnt1 = 73
          ev_cnt3_int = 5
          ev_cnt4_float = 5.431
          values_to_write(1) = 9


          be_verbose = 0
          if(command_argument_count() .eq. 1) then
              call get_command_argument(number=1, length=arglen, status=stat)
              if( stat .eq. 0 ) then
                  allocate (character(arglen) :: arg)
                  call get_command_argument(number=1, value=arg, status=stat)
                  if( arg == '-verbose' ) then
                      be_verbose = 1
                  endif
              endif
          endif


          call papif_sde_init('TESTLIB', handle, error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_init'
          call papif_sde_register_counter(handle, 'TESTEVENT', PAPI_SDE_RO, PAPI_SDE_long_long, C_loc(ev_cnt1), error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_register_counter'
          call papif_sde_describe_counter(handle, 'TESTEVENT', 'This is a test counter used to test SDE from FORTRAN, for testing purposes only. Use it when you test the functionality in a test or something. Happy testing.', error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_describe_counter'
          call papif_sde_register_counter(handle, 'SERIOUSEVENT', PAPI_SDE_RO, PAPI_SDE_long_long, C_loc(ev_cnt2), error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_register_counter'
          ! The following call should be ignored by the SDE component (since this counter is already registered.)
          call papif_sde_register_counter(handle, 'SERIOUSEVENT', PAPI_SDE_RO, PAPI_SDE_long_long, C_loc(ev_cnt1), error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_register_counter'
          call papif_sde_describe_counter(handle, 'SERIOUSEVENT', 'This is a not a test counter, this one is serious.', error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_describe_counter'
          call papif_sde_register_counter(handle, 'FLOATEVENT', PAPI_SDE_RO, PAPI_SDE_float, C_loc(ev_cnt4_float), error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_register_counter'

          internal_variable = 987.65
          internal_variable_int = 12345

          ! the following call should be ignored by the SDE component, but the returned 'handle' should still be valid.
          call papif_sde_init('TESTLIB', handle, error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_init'
          call papif_sde_register_counter_cb(handle, 'FP_EVENT', PAPI_SDE_RO, PAPI_SDE_long_long, c_funloc(f08_callback), C_loc(internal_variable), error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_register_counter_cb'
          call papif_sde_describe_counter(handle, 'FP_EVENT', 'This is another counter.', error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_describe_counter'
          ! The following call should be ignored by the SDE component (since this counter is already registered.)
          call papif_sde_register_counter_cb(handle, 'FP_EVENT', PAPI_SDE_RO, PAPI_SDE_long_long, c_funloc(f08_callback), C_loc(ev_cnt1), error)
          if(error .ne. PAPI_OK ) print *,'Error in sde_register_counter_cb'

          call xandria_init()
          call gamum_init()
          call recorder_init()

          internal_variable = 11.0

          ret_val = PAPI_VER_CURRENT

          call papif_library_init(ret_val)
          if( ret_val .ne. PAPI_VER_CURRENT ) then
              print *,'Error at papif_init', ret_val, '!=', PAPI_VER_CURRENT
              print *,'PAPI_EINVAL', PAPI_EINVAL
              print *,'PAPI_ENOMEM', PAPI_ENOMEM
              print *,'PAPI_ECMP', PAPI_ECMP
              print *,'PAPI_ESYS', PAPI_ESYS
              stop
          endif

          call recorder_do_work()

          eventset = PAPI_NULL
          call papif_create_eventset( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_create_eventset'
              stop
          endif

          eventset2 = PAPI_NULL
          call papif_create_eventset( eventset2, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_create_eventset'
              stop
          endif

! 1
          call papif_event_name_to_code( 'sde:::TESTLIB::TESTEVENT', eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

!-------------------------------------------------------------------------------

          call recorder_do_work()

!-------------------------------------------------------------------------------

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          ev_cnt1 = ev_cnt1+100
          call xandria_do_work()

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif
    
          call recorder_do_work()

          if( be_verbose .eq. 1 ) print '(A29,I4)',   '  TESTLIB::TESTEVENT (100) = ', values(1)
          if( values(1) .ne. 100 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

! 2
          call papif_event_name_to_code( 'sde:::TESTLIB::FP_EVENT', eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

          internal_variable = 12.0
          internal_variable_int = 12

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          ev_cnt1 = ev_cnt1+9
          ev_cnt4_float = ev_cnt4_float+33
          internal_variable = 12.4

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A27,I2)',   '  TESTLIB::TESTEVENT (9) = ', values(1)
          if( values(1) .ne. 9 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A26,I2)',   '  TESTLIB::FP_EVENT (0) = ', values(2)
          if( values(2) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

! 3
          call papif_event_name_to_code( 'sde:::TESTLIB::FLOATEVENT', eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call recorder_do_work()

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

! 4
          call papif_event_name_to_code( 'sde:::Xandria::EV_I1', eventcode, ret_val )
! not added
          call papif_event_name_to_code( 'sde:::Xandria::EV_I2', junk, ret_val )
! not added
          call papif_event_name_to_code( 'sde:::Xandria::EV_I2', junk, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

! 5
          call papif_add_named_event( eventset, 'sde:::Xandria::RW_I1', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

          do i=1,37
              call recorder_do_work()
          end do

!-------------------------------------------------------------------------------
          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          ev_cnt1 = ev_cnt1+2
          ev_cnt4_float = ev_cnt4_float+3.98
          internal_variable = 20.12
          internal_variable_int = 20

          call xandria_do_work()

! Adding the 5th counter into a separate eventset so we can write into it.
          call papif_add_named_event( eventset2, 'sde:::Xandria::RW_I1', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

!--------------------
          if( be_verbose .eq. 1 ) print *,''

          call papif_read(eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_read'
              stop
          endif

          do i=1,370
              call recorder_do_work()
          end do

          if( be_verbose .eq. 1 ) print '(A27,I2)',   '  TESTLIB::TESTEVENT (2) = ', values(1)
          if( values(1) .ne. 2 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A26,I2)',   '  TESTLIB::FP_EVENT (8) = ', values(2)
          if( values(2) .ne. 8 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(3), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A31,F4.2)', '  TESTLIB::FLOATEVENT (3.98) = ', value_d
          if( abs(value_d - 3.98) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A23,I1)',   '  Xandria::EV_I1 (1) = ', values(4)
          if( values(4) .ne. 1 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::RW_I1 (14) = ', values(5)
          if( values(5) .ne. 14 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          call xandria_do_work()
          call xandria_do_work()
          call xandria_do_work()

!--------------------
          if( be_verbose .eq. 1 ) print *,''

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::RW_I1 (35) = ', values(5)
          if( values(5) .ne. 35 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

!-------------------------------------------------------------------------------
! WRITE and then read the RW counter.
          call papif_start( eventset2, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call papif_write(eventset2, values_to_write, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_write'
              stop
          endif

          call papif_read(eventset2, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_read'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A23,I1)',   '  Xandria::RW_I1 (9) = ', values(1)
          if( values(1) .ne. 9 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          call papif_stop( eventset2, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif
!-------------------------------------------------------------------------------

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          ev_cnt1 = ev_cnt1+5
          ev_cnt4_float = ev_cnt4_float+18.8
          internal_variable = internal_variable + 30.1
          internal_variable_int = 30

          call xandria_do_work()
          call xandria_do_work()
          call xandria_do_work()

          call papif_read(eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_read'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A27,I1)',  '  TESTLIB::TESTEVENT (5) = ', values(1)
          if( values(1) .ne. 5 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A27,I2)',  '  TESTLIB::FP_EVENT (30) = ', values(2)
          if( values(2) .ne. 30 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(3), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A31,F4.1)','  TESTLIB::FLOATEVENT (18.8) = ', value_d
          if( abs(value_d - 18.8) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A23,I2)',  '  Xandria::EV_I1 (3) = ', values(4)
          if( values(4) .ne. 3 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif


!--------------------
          if( be_verbose .eq. 1 ) print *,''

          call papif_reset(eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_reset'
              stop
          endif

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif


          if( be_verbose .eq. 1 ) print '(A27,I2)',   '  TESTLIB::TESTEVENT (0) = ', values(1)
          if( values(1) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A26,I2)',   '  TESTLIB::FP_EVENT (0) = ', values(2)
          if( values(2) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(3), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A31,F4.1)', '  TESTLIB::FLOATEVENT, (0.0) = ', value_d
          if( abs(value_d - 0.0) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A23,I2)',   '  Xandria::EV_I1 (0) = ', values(4)
          if( values(4) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

! 6
          call papif_event_name_to_code('sde:::Xandria::EV_R1' , codes(1), ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

! 7
          call papif_event_name_to_code('sde:::Xandria::EV_R2' , codes(2), ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

! 8
          call papif_event_name_to_code('sde:::Xandria::EV_R3' , codes(3), ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_events( eventset, codes, 3, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_events'
              stop
          endif


          do i=1,29
              call recorder_do_work()
          end do

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call xandria_do_work()
          call xandria_do_work()
          call xandria_do_work()

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A27,I2)',   '  TESTLIB::TESTEVENT (0) = ', values(1)
          if( values(1) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A26,I2)',   '  TESTLIB::FP_EVENT (0) = ', values(2)
          if( values(2) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(3), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A30,F3.1)' ,'  TESTLIB::FLOATEVENT (0.0) = ', value_d
          if( abs(value_d - 0.0) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A23,I2)',   '  Xandria::EV_I1 (3) = ', values(4)
          if( values(4) .ne. 3 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::EV_R1 (30) = ', values(6)
          if( values(6) .ne. 30 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::EV_R2 (60) = ', values(7)
          if( values(7) .ne. 60 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::EV_R3 (90) = ', values(8)
          if( values(8) .ne. 90 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

!-------------------------------------------------------------------------------
          call gamum_unreg()

          do i=1,248
              call recorder_do_work()
          end do

! 9
          call papif_event_name_to_code('sde:::Gamum::ev1' , codes(1), ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

! 10
          call papif_event_name_to_code('sde:::Gamum::ev3' , codes(2), ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

! 11
          call papif_event_name_to_code('sde:::Gamum::ev4' , codes(3), ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_events( eventset, codes, 3, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_events'
              stop
          endif

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call gamum_do_work()
          call gamum_do_work()
          call gamum_do_work()
          call gamum_do_work()

          do i=1,122
              call recorder_do_work()
          end do

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif


          value_d = transfer(values(9), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A21,F4.1)','  Gamum::ev1 (0.4) = ', value_d
          if( abs(value_d - 0.4) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(10), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A21,F4.1)','  Gamum::ev3 (0.8) = ', value_d
          if( abs(value_d - 0.8) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(11), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A23,F5.3)','  Gamum::ev4 (1.888) = ', value_d
          if( abs(value_d - 1.888) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

! 12
          call papif_event_name_to_code('sde:::Xandria::LATE' , eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

! We register this event after the placeholder was created
          call xandria_add_more()

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call xandria_do_work()

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          value_d = transfer(values(9), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A21,F4.1)',     '  Gamum::ev1 (0.0) = ', value_d
          if( abs(value_d - 0.0) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(10), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A21,F4.1)','  Gamum::ev3 (0.0) = ', value_d
          if( abs(value_d - 0.0) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(11), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A23,F5.3)',     '  Gamum::ev4 (1.888) = ', value_d
          if( abs(value_d - 1.888) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A22,I2)',   '  Xandria::LATE (7) = ', values(12)
          if( values(12) .ne. 7 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif


          do i=1,9
              call recorder_do_work()
          end do

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

! 13
          call papif_event_name_to_code('sde:::Xandria::WRONG' , eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call xandria_do_work()
          call xandria_do_work()
          call xandria_do_work()

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A23,I2)',   '  Xandria::LATE (21) = ', values(12)
          if( values(12) .ne. 21 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::WRONG (-1) = ', values(13)
          if( values(13) .ne. -1 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''
! 14
          call papif_event_name_to_code('sde:::Gamum::group0' , eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

! 15
          call papif_event_name_to_code('sde:::Gamum::papi_counter' , eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call gamum_do_work()
          call gamum_do_work()

          call papif_read(eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_read'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A22,I2)',   '  Xandria::LATE (0) = ', values(12)
          if( values(12) .ne. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A24,I2)',   '  Xandria::WRONG (-1) = ', values(13)
          if( values(13) .ne. -1 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(9), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A21,F4.1)',     '  Gamum::ev1 (0.2) = ', value_d
          if( abs(value_d - 0.2) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(11), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A23,F5.3)',     '  Gamum::ev4 (2.332) = ', value_d
          if( abs(value_d - 2.332) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(14), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A36,F5.3)',     '  Gamum::group0 [ev1+ev4] (2.532) = ', value_d
          if( abs(value_d - 2.532) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A29,I3)',     '  Gamum::papi_counter (36) = ', values(15)
          if( abs(values(15) - 36) .gt. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          do i=1,5
              call gamum_do_work()
          end do
          call xandria_do_work()
          do i=1,217
              call recorder_do_work()
          end do

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          value_d = transfer(values(9), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A21,F3.1)',     '  Gamum::ev1 (0.7) = ', value_d
          if( abs(value_d - 0.7) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(11), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A23,F5.3)',     '  Gamum::ev4 (3.442) = ', value_d
          if( abs(value_d - 3.442) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          value_d = transfer(values(14), 1.0D0)
          if( be_verbose .eq. 1 ) print '(A36,F5.3)',     '  Gamum::group0 [ev1+ev4] (4.142) = ', value_d
          if( abs(value_d - 4.142) .gt. rounding_error(value_d) ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

          if( be_verbose .eq. 1 ) print '(A29,I3)',     '  Gamum::papi_counter (66) = ', values(15)
          if( abs(values(15) - 66) .gt. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

! 16
          call papif_add_named_event(eventset, 'sde:::Lib_With_Recorder::simple_recording:CNT', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

! 17
          call papif_add_named_event(eventset, 'sde:::Lib_With_Recorder::simple_recording:MIN', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

! 18
          call papif_add_named_event(eventset, 'sde:::Lib_With_Recorder::simple_recording:Q1', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

! 19
          call papif_add_named_event(eventset, 'sde:::Lib_With_Recorder::simple_recording:MED', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

! 20
          call papif_add_named_event(eventset, 'sde:::Lib_With_Recorder::simple_recording:Q3', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

! 21
          call papif_add_named_event(eventset, 'sde:::Lib_With_Recorder::simple_recording:MAX', ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_named_event'
              stop
          endif

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A51,I4)',     '  Lib_With_Recorder::simple_recording:CNT (1036) = ', values(16)
          if( abs(values(16) - 1036) .gt. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif
          
          call c_f_pointer(transfer(values(17), quantile), quantile_f)
          if( be_verbose .eq. 1 ) print '(A54,I6)',     '  Lib_With_Recorder::simple_recording:MIN (     >0) = ', quantile_f

          call c_f_pointer(transfer(values(18), quantile), quantile_f)
          if( be_verbose .eq. 1 ) print '(A54,I6)',     '  Lib_With_Recorder::simple_recording:Q1  ( ~30864) = ', quantile_f

          call c_f_pointer(transfer(values(19), quantile), quantile_f)
          if( be_verbose .eq. 1 ) print '(A54,I6)',     '  Lib_With_Recorder::simple_recording:MED ( ~61728) = ', quantile_f

          call c_f_pointer(transfer(values(20), quantile), quantile_f)
          if( be_verbose .eq. 1 ) print '(A54,I6)',     '  Lib_With_Recorder::simple_recording:Q3  ( ~92592) = ', quantile_f

          call c_f_pointer(transfer(values(21), quantile), quantile_f)
          if( be_verbose .eq. 1 ) print '(A54,I6)',     '  Lib_With_Recorder::simple_recording:MAX (<123456) = ', quantile_f

!-------------------------------------------------------------------------------
          if( be_verbose .eq. 1 ) print *,''

! 22
          call papif_event_name_to_code('sde:::Xandria::XND_CREATED' , eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_name_to_code'
              stop
          endif

          call papif_add_event( eventset, eventcode, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_add_event'
              stop
          endif

          call papif_start( eventset, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_start'
              stop
          endif

          call xandria_do_work()
          call xandria_do_work()
          call xandria_do_work()

          call papif_stop( eventset, values, ret_val )
          if( ret_val .ne. PAPI_OK ) then
              print *,'Error at papif_stop'
              stop
          endif

          if( be_verbose .eq. 1 ) print '(A30,I2)',   '  Xandria::XND_CREATED (27) = ', values(22)
          if( abs(values(22) - 27) .gt. 0 ) then
              if( be_verbose .eq. 1 ) print *,'^^^^^^^^^^^^^^^^^^^'
              all_tests_passed = 0
          endif
          
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

          call papif_shutdown( )

          if( be_verbose .eq. 1 ) print *,''
          if( all_tests_passed .eq. 1 ) then
              call ftests_pass('')
          else
              call ftest_fail(__FILE__, __LINE__, 'SDE counters do not much expected values!', 1)
          endif



        end program

        function rounding_error(param) result(ret_val)
            real (KIND=KIND(0.0D0)) :: param, ret_val

            ret_val = param/100000.0
        end function rounding_error

        function f08_callback(param) result(ret_val)
          use, intrinsic :: ISO_C_BINDING
          implicit none
          real :: param
          integer(kind=C_LONG_LONG) :: ret_val

          ret_val = int(param, C_LONG_LONG)
        end function f08_callback

