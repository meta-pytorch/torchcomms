#include "fpapi_test.h"

      program second
      implicit integer (p)
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer*8 values(10), max, min
      integer es1, es2, es3
      integer retval

      Integer last_char
      External last_char
      integer tests_quiet, get_quiet
      external get_quiet
#if (defined(sgi) && defined(host_mips))
      integer id 
      integer*4 getuid
#endif

#if (defined(sgi) && defined(host_mips))
      id = getuid()
#endif

      tests_quiet = get_quiet()
      es1 = PAPI_NULL
      es2 = PAPI_NULL
      es3 = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_query_event(PAPI_TOT_INS, retval)
      if (retval.NE.PAPI_OK) then
        call ftest_skip(__FILE__, __LINE__, 'PAPI_FP_INS', PAPI_ENOEVNT)
      end if

      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es1, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if
      
      call PAPIf_add_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if

      call PAPIf_create_eventset(es2, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es2, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if
      
      call PAPIf_add_event( es2, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if

      call PAPIf_create_eventset(es3, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es3, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if
      
      call PAPIf_add_event( es3, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if

      call PAPIf_set_event_domain(es1, PAPI_DOM_ALL, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_set_domain', retval)
      end if

      call PAPIf_set_event_domain(es2, PAPI_DOM_KERNEL, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_set_domain', retval)
      end if

      call PAPIf_set_event_domain(es3, PAPI_DOM_USER, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_set_domain', retval)
      end if

      call PAPIf_start(es1, retval)

      call fdo_flops(NUM_FLOPS)

      if (retval.eq.PAPI_OK) then
      call PAPIf_stop(es1, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if
      end if

      call PAPIf_start(es2, retval)

      call fdo_flops(NUM_FLOPS)

      if (retval.eq.PAPI_OK) then
      call PAPIf_stop(es2, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if
      end if

      call PAPIf_start(es3, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_start', retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_stop(es3, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if

      call PAPIf_remove_event( es1, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      call PAPIf_remove_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      call PAPIf_remove_event( es2, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      call PAPIf_remove_event( es2, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      call PAPIf_remove_event( es3, PAPI_TOT_INS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if
      call PAPIf_remove_event( es3, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, 'Test case 2: Non-overlapping start, stop, read',
     *' for all 3 domains.'
      print *, '-------------------------------------------------'//
     *  '------------------------------'
      end if

      call PAPIf_get_domain(es1, domain, PAPI_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) 'Default domain is:', domain, domainstr
      end if
 900  format(a20, i3, ' ', a70)

      call PAPIf_get_granularity(es1, granularity, PAPI_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_get_granularity', 
     *retval)
      end if

      call stringify_granularity(granularity, grnstr)
      if (tests_quiet .EQ. 0) then
      write (*,800) 'Default granularity is:', granularity, grnstr
      end if
 800  format(a25, i3, ' ', a20)

      if (tests_quiet .EQ. 0) then
      print *, 'Using', NUM_FLOPS, ' iterations of c += b*c'

      print *, '-------------------------------------------------'//
     *  '------------------------------'

      print *, 'Test type   :       PAPI_DOM_ALL    PAPI_DOM_KERNEL',
     *'   PAPI_DOM_USER'
      write (*,200) 'PAPI_TOT_INS', values(1), values(3), values(5)
      write (*,200) 'PAPI_TOT_CYC', values(2), values(4), values(6)
 200  format(A15, ': ', I15, I15, I15)

      print *, '-------------------------------------------------'//
     *  '------------------------------'

      print *, 'Verification:'
      print *, 'Row 1 approximately equals N 0 N'
      print *, 'Column 1 approximately equals column 2 plus column 3'
#if defined(sgi) && defined(host_mips)
      print *
      print *, '* IRIX requires root for PAPI_DOM_KERNEL',
     *' and PAPI_DOM_ALL.'
      print *, '* The first two columns will be invalid if not',
     *' run as root for IRIX.'
#endif
      end if

#if (defined(sgi) && defined(host_mips))
      if (id.NE.0) then
         min = NUM_FLOPS*0.9
         max = NUM_FLOPS*1.1
         if ((values(5) .lt. min) .OR. (values(5) .gt. max)) then
            call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_FP_INS', 1)
         end if
      else
         min = values(5)*0.9
         max = values(5)*1.1
         if ((values(1) .lt. min) .OR. (values(1) .gt. max)) then
            call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_FP_INS', 1)
         end if            
         min = values(2)*0.9
         max = values(2)*1.1
         if (((values(4)+values(6)) .lt. min) .OR. 
     *       ((values(4)+values(6)) .gt. max)) then
            call ftest_fail(__FILE__, __LINE__, 'PAPI_TOT_CYC', 1)
         end if            
      endif
#else
      min = INT(REAL(values(5))*0.9)
      max = INT(REAL(values(5))*1.1)
      if ((values(1) .lt. min) .OR. (values(1) .gt. max)) then
         call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_FP_INS', 1)
      end if            

      min = INT(REAL(values(2))*0.8)
      max = INT(REAL(values(2))*1.2)
      if (((values(4)+values(6)) .lt. min) .OR. 
     *    ((values(4)+values(6)) .gt. max)) then
         call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_TOT_CYC', 1)
      end if            
#endif

      call ftests_pass(__FILE__)
      end
