#include "fpapi_test.h"

#define MAX_TO_ADD 5

      program multiplex2
      IMPLICIT integer (p)

      integer retval
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

      if (tests_quiet .EQ. 0) then
        write (*, 100) NUM_ITERS
 100    FORMAT ("multiplex2: Using ", I3, " iterations")
        write (*,*) "case1: Does PAPI_multiplex_init() handle", 
     *       " lots of events?"
      end if
      call case1(tests_quiet, retval)
      call ftests_pass(__FILE__)
      end

      subroutine init_papi()
      IMPLICIT integer (p)
      integer retval

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPI_library_init', retval)
      end if
      end

      subroutine case1(tests_quiet, ret)
      IMPLICIT integer (p)
      integer tests_quiet
      integer retval
      integer i, ret, fd
      integer EventCode
      character*(PAPI_MAX_STR_LEN) event_name, event_descr,
     *     event_label, event_note
      integer avail_flag, flags, check
      integer EventSet,mask1
      integer*8 values(MAX_TO_ADD*2)

      EventSet = PAPI_NULL
      call init_papi()

      call init_multiplex()

      call PAPIf_create_eventset(EventSet, retval)
      if ( retval.NE.PAPI_OK) then

        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_create_eventset',
     *       retval)
      end if

      call PAPIf_assign_eventset_component(EventSet, 0, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     &      'PAPIf_assign_eventset_component', retval)
      end if

      call PAPIf_set_multiplex(EventSet, retval)
      if ( retval.EQ.PAPI_ENOSUPP) then
        call ftest_skip(__FILE__, __LINE__,
     .       'Multiplex not implemented', retval)
      end if
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .       'papif_set_multiplex', retval)
      end if

      if (tests_quiet .EQ. 0) then
        print *, "Checking for available events..."
      end if

      EventCode = 0
      i = 1
      do while (i .LE. MAX_TO_ADD)
         avail_flag=0
         do while ((avail_flag.EQ.0).AND.
     *        (EventCode.LT.PAPI_MAX_PRESET_EVENTS))
              mask1 = ((PAPI_L1_DCM)+EventCode) 
          if (mask1.NE.PAPI_TOT_CYC) then
              call papif_get_event_info(mask1, 
     *          event_name, event_descr, event_label, avail_flag, 
     *          event_note, flags, check)
          end if
          EventCode = EventCode + 1
         end do

         if ( EventCode.EQ.PAPI_MAX_PRESET_EVENTS .AND.
     *        i .LT. MAX_TO_ADD ) then
            call ftest_fail(__FILE__, __LINE__,
     *           'PAPIf_add_event', retval)
         end if

         if (tests_quiet .EQ. 0) then
            write (*, 200) "  Adding Event ", event_name 
 200        FORMAT(A22, A12)
         end if

         mask1 = ((PAPI_L1_DCM)+EventCode) 
         mask1 = mask1 - 1
         call PAPIf_add_event( EventSet, mask1, retval )
         if ( retval .NE. PAPI_OK .AND. retval .NE. PAPI_ECNFLCT) then
            call ftest_fail(__FILE__, __LINE__,
     *           'PAPIf_add_event', retval)
            stop
         end if

         if (tests_quiet .EQ. 0) then
         if (retval .EQ. PAPI_OK) then
             write (*, 200) "  Added Event ", event_name
         else
             write (*, 200) "  Could not add Event ", event_name
           end if
         end if
      
       if (retval .EQ. PAPI_OK) then
         i = i + 1
       end if
      end do
      
      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_start', retval)
      end if

      fd = 1
      call do_stuff()

      call PAPIf_stop(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_stop', retval)
      end if

      call PAPIf_cleanup_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_cleanup_eventset',
     *       retval)
      end if

      call PAPIf_destroy_eventset(EventSet, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_destroy_eventset',
     *       retval)
      end if

      ret = SUCCESS

      end

