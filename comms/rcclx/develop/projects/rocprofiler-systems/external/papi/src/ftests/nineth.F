#include "fpapi_test.h"

      program nineth
      implicit integer (p)

      integer es1, es2
      integer*8 values(10),tvalues(10)
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      integer retval
      integer clockrate
      real*8 test_flops, min, max

      Integer last_char
      External last_char
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_query_event(PAPI_FP_OPS, retval)
      if (retval.NE.PAPI_OK) then
        call ftest_skip(__FILE__, __LINE__, 'PAPI_FP_OPS', PAPI_ENOEVNT)
      end if

      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es1, PAPI_FP_OPS, retval )
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

      call PAPIf_add_event( es2, PAPI_FLOPS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if

      call PAPIf_get_clockrate(clockrate)
      if (tests_quiet .EQ. 0) then
        print *, 'Clockrate:', clockrate
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es1, tvalues(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if

      call PAPIf_start(es2, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_start', retval)
      end if

      call do_flops(NUM_FLOPS)

      call PAPIf_stop(es2, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if

      call PAPIf_remove_event( es1, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      call PAPIf_remove_event( es1, PAPI_FP_OPS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      call PAPIf_remove_event( es2, PAPI_FLOPS, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_remove_event', retval)
      end if

      test_flops = tvalues(1)*clockrate*1000000.0
      if ( tvalues(2) .NE. 0) then
          test_flops = test_flops / tvalues(2)
      else
          test_flops = 0.0
      end if
    
      if (tests_quiet .EQ. 0) then
      print *, "Test case 9: start, stop for derived event PAPI_FLOPS"
      print *, "---------------------------------------------"
      end if
      call PAPIf_get_domain(es1, domain, PAPI_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) "Default domain is:", domain, domainstr
 900  format(a20, i3, " ", a70)
      end if

      call PAPIf_get_granularity(es1, granularity, PAPI_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_get_granularity', 
     *retval)
      end if

      call stringify_granularity(granularity, grnstr)
      if (tests_quiet .EQ. 0) then
      write (*,800) "Default granularity is:", granularity, grnstr
 800  format(a25, i3, " ", a20)

      print *, " Using", NUM_FLOPS, " iterations of c += b*c"
      print *, "---------------------------------------------"
 
      write (*,810) "Test type   :", 1, 2
      write (*,810) "PAPI_FP_OPS :", tvalues(1), 0
      write (*,810) "PAPI_TOT_CYC:", tvalues(2), 0
      write (*,810) "PAPI_FLOPS  :", 0, values(1)
      print *, "---------------------------------------------"      
 810  format(a15, i15, i15)

      print *, "Verification:"
      print *, "Last number in row 3 approximately equals", test_flops
      end if

      min = values(1) * 0.9
      max = values(1) * 1.1
      if ((test_flops.gt.max) .OR. (test_flops.lt.min)) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_FLOPS', 1)
      end if       

      call ftests_pass(__FILE__)
      end
