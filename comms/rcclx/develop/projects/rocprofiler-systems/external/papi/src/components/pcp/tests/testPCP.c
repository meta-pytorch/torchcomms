//----------------------------------------------------------------------------
// This test program exercises the functions within the linux-pcp.c component.
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include "papi_test.h" 
#include <sys/time.h>

#define mConvertUsec(timeval_) ((double) (timeval_.tv_sec*1000000+timeval_.tv_usec))     /* avoid typos, make it a double. */
static struct timeval t1, t2;

unsigned long dummyThreadId(void) { return(0); }                        // dummy return to give a thread id.

typedef union
{
   long long ll;
   unsigned long long ull;
   double    d;
   void *vp;
   unsigned char ch[8];
} convert_64_t;

//-----------------------------------------------------------------------------
// MAIN.
//-----------------------------------------------------------------------------

int main(int argc, char **argv) {                                       // args to allow quiet flags. 
   int i,k,m,code=0;
   int eventSetCount, ret;
   PAPI_event_info_t info;
   int cid=-1;                                                          // signal for not found.
   int EventSet = PAPI_NULL;
	int quiet=0;
   char errMsg[1024];                                                   // space for an error message with more info.
	quiet=tests_quiet( argc, argv );                                     // From papi_test.h.
   int firstTime = 1;                                                   // some things we want to run once.
   int retlen;

   gettimeofday(&t1, NULL);
   ret = PAPI_library_init( PAPI_VER_CURRENT );
   gettimeofday(&t2, NULL);
   if (!quiet) fprintf(stderr, "PAPI_Library_init run time = %9.1f uS\n", (mConvertUsec(t2)-mConvertUsec(t1)));
 
   if (ret != PAPI_VER_CURRENT) {                                       // if we failed, 
      test_fail(__FILE__, __LINE__, "PAPI_library_init failed\n", ret); // report.
   }


	if (!quiet) {
	   fprintf(stderr, "Testing PCP Component with PAPI %d.%d.%d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	}

   // PAPI_init_thread should be run only once, immediately after 
   // library init. 
// ret = PAPI_thread_init(dummyThreadId);                               // PCP doesn't do anything, but should not err or crash on thread init.
// if (ret != PAPI_OK) {                                                // If we get an error, this tester needs updating; pcp code has changed.
//    test_fail(__FILE__, __LINE__, "PAPI_thread_init failed; this tester needs to be updated.\n", ret); // report.
// }
   
   // Find our component, pcp; 
   PAPI_component_info_t *aComponent;
   k = PAPI_num_components();                                           // get number of components.
   for (i=0; i<k && cid<0; i++) {                                       // while not found,
      aComponent = (PAPI_component_info_t*) PAPI_get_component_info(i); // get the component info.
      if (aComponent == NULL) {                                         // if we failed,
         sprintf(errMsg,  "PAPI_get_component_info(%i) failed, "
            "returned NULL. %i components reported.\n", i,k);
         test_fail(__FILE__, __LINE__, errMsg, 0);

      }

      if (strcmp("pcp", aComponent->name) == 0) cid=i;                  // If we found our match, record it.
   } // end search components.

   if (cid < 0) {                                                       // if no PCP component found,
      sprintf(errMsg, "Failed to find pcp component among %i "
            "reported components.\n", k);
      test_fail(__FILE__, __LINE__, errMsg, 0);                         // report it.
   }

	if (!quiet) {
	  fprintf(stderr, "Found PCP Component at id %d\n",cid);
	}

   // If the component is disabled, skip the test.
   if (aComponent->disabled) {
       test_skip(__FILE__,__LINE__,"Component pcp is disabled", 0);
       PAPI_shutdown();
       return 0;
   }

   // Library is initialized and pcp component is identified.

   // Set up to exercise the code for _pcp_ctl; it does nothing 
   // but prove it doesn't crash if called. The actual call to
   // PAPI_set_opt is below; it requires an eventset. 

   PAPI_option_t aDomOpt;                                            // make a domain option.
   aDomOpt.domain.def_cidx = cid;                                    // fill it out.
   aDomOpt.domain.domain = PAPI_DOM_ALL;                             // .. 

   // Begin enumeration of all events. 

   m=PAPI_NATIVE_MASK;                                               // Get the PAPI NATIVE mask.
   ret=PAPI_enum_cmp_event(&m,PAPI_ENUM_FIRST,cid);                  // Begin enumeration of ALL papi counters.
   if (ret != PAPI_OK) fprintf(stderr, "PAPI_enum_cmp_event returned %i [%s].\n", ret, PAPI_strerror(ret));
   if (ret != PAPI_OK) {                                             // If that failed, report and exit.
      test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event failed.\n",
         ret);
   }

   //----------------------------------------------
   // Setups are done, we begin the work work here.
   //----------------------------------------------
   int count = 0;
   if (!quiet) printf("Component Idx, Symbol, Units, Description, HexCode (this run only), Time Scope, PAPI_TYPE, Sample Value\n");

   do{                                                               // Enumerate all events.
      memset(&info,0,sizeof(PAPI_event_info_t));                     // Clear event info.
      k=m;                                                           // Make a copy of current code.

      // enumerate sub-events, with masks. For this test, we do not
      // have any! But we do this to test our enumeration works as
      // expected. First time through is guaranteed, of course.

      do{                                                            // enumerate masked events. 
         ret=PAPI_get_event_info(k,&info);                           // get name of k symbol.
         if (ret != PAPI_OK) {                                       // If that failed, report and exit.
            sprintf(errMsg, "PAPI_get_event_info(%i) failed.\n", k); // build message.
            test_fail(__FILE__, __LINE__, errMsg, ret);
         }

         // Test creating an event set.
         ret = PAPI_create_eventset(&EventSet);                      // Try it.
         if (ret != PAPI_OK) {                                       // If that failed, report and exit.
         test_fail(__FILE__, __LINE__, "PAPI_enum_create_eventset failed.\n", ret);
         }
         
         // Test adding and removing the event by name.
         ret=PAPI_add_named_event(EventSet,info.symbol);             // Try to add it for counting.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_add_named_event('%s') failed.\n", info.symbol);
            if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
            test_fail( __FILE__, __LINE__, errMsg, ret);
         }

         // test _pcp_ctl function. Just need to do this once.

         if (firstTime) {
            aDomOpt.domain.eventset = EventSet;                         // ..
            ret = PAPI_set_opt(PAPI_DOMAIN, &aDomOpt);                  // force call to pcp_ctl.
            if (ret != PAPI_OK) {                                       // If that failed, report and exit.
               test_fail(__FILE__, __LINE__, "PAPI_set_opt failed.\n",
               ret);
            } 
         }   

         ret=PAPI_remove_named_event(EventSet,info.symbol);          // Try to remove it.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_remove_named_event('%s') failed.\n", info.symbol);
            if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
            test_fail( __FILE__, __LINE__, errMsg, ret);
         }

         // Test getting code for name, consistency with enumeration.
         ret=PAPI_event_name_to_code(info.symbol, &code);            // Try to read the code from the name.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_event_name_to_code('%s') failed.\n", info.symbol);
            if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
            test_fail( __FILE__, __LINE__, errMsg, ret);
         }
   
         // Papi can report a different code; k incremented by 1.
         // I am not clear on why it does that.
         if (code != k && code != (k+1)) {                           // If code retrieved is not the same, fail and report it.
            retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_event_name_to_code('%s') "
               "returned code 0x%08X, expected 0x%08X. failure.\n", 
               info.symbol, code, k);
            if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
            test_fail( __FILE__, __LINE__, errMsg, 0);               // report and fail.  
         }

         // Test getting name from code, consistency with info.
         char testName[PAPI_MAX_STR_LEN] = "";                       // needed for test.
         ret=PAPI_event_code_to_name(code, testName);                // turn code back into a name.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_event_code_to_name(('0x%08X') failed.\n", code);
            if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
            test_fail( __FILE__, __LINE__, errMsg, ret);
         }
   
         if (strcmp(info.symbol, testName) != 0) {                   // If name retrieved is not the same, fail and report it.
            retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_event_code_to_name(('0x%08X') "
               "returned name=\"'%s'\", expected \"%s\". failure.\n", 
               code, testName, info.symbol);
            if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
            test_fail( __FILE__, __LINE__, errMsg, 0);               // Report and exit.
         }

         ret = PAPI_add_event(EventSet, code);                       // Try to add the event.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            sprintf(errMsg, "PAPI_add_event('0x%08X') failed.\n", code);
            test_fail( __FILE__, __LINE__, errMsg, ret);             // report and exit.
         }

         ret = PAPI_start(EventSet);                                 // start counting.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            sprintf(errMsg, "PAPI_start_event('0x%08X') failed.\n", code);
            test_fail( __FILE__, __LINE__, errMsg, ret);             // report and exit.
         }
         
         long long *values = NULL;                                   // pointer for us to malloc next.

         eventSetCount = PAPI_num_events(EventSet);                  // get the number of events in set.
         if (eventSetCount < 1) {
            test_fail( __FILE__, __LINE__, "PAPI_num_events(EventSet) failed.\n", ret);
         }

         values = calloc(eventSetCount, sizeof(long long));          // make zeroed space for it. 
         
         ret = PAPI_read(EventSet, values);                          // read without a stop.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            free(values);
            test_fail( __FILE__, __LINE__, "PAPI_read(EventSet) failed.\n", ret);
         }

         // Test doing something with it.
         count++;                                                    // bump count of items seen and added.
         if (!quiet) {
            printf("%i, %s, %s, %s, 0x%08x,", 
            info.component_index, info.symbol,info.units, 
            info.long_descr, info.event_code);
            convert_64_t cvt;
            cvt.ll = values[0];                                      // copy the value returned.

            if (info.timescope == PAPI_TIMESCOPE_SINCE_START) { 
               printf("SINCE START,");
            } else {
               printf("POINT,"); 
            }
                  
            switch (info.data_type) {                                // based on type, 
               case PAPI_DATATYPE_INT64:
                  printf("INT64, %lli", cvt.ll);
                  break;                                             // END CASE.

               case PAPI_DATATYPE_UINT64:
                  printf("UINT64, %llu", cvt.ull);
                  break;                                             // END CASE.

               case PAPI_DATATYPE_FP64:
                  printf("FP64, %f", cvt.d);
                  break;                                             // END CASE.
            
               default:
                  printf("UNKNOWN TYPE, %p", cvt.vp);
                  break;                                             // END CASE.
            }

            printf("\n");   
         }

         ret = PAPI_reset(EventSet);                                 // Reset the event.
         if (ret != PAPI_OK) {                                       // If that failed, report and exit.
            free(values);
            test_fail( __FILE__, __LINE__, "PAPI_reset_event() failed\n", ret);
         }

         ret = PAPI_stop(EventSet, values);                          // stop counting, get final values.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            free(values);
            test_fail( __FILE__, __LINE__, "PAPI_stop_event(EventSet, values) failed.\n", ret);
         }

         free(values);                                               // free alloc memory.
         values = NULL;                                              // prevent double free.

         ret = PAPI_cleanup_eventset(EventSet);                      // Try a cleanup.
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset(EventSet) failed.\n", ret);
         }

         ret = PAPI_destroy_eventset(&EventSet);                           // Deallocate. No memory leaks!
         if (ret != PAPI_OK) {                                       // If that failed, report it.
            test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset(EventSet) failed.\n", ret);
         }

         firstTime = 0;                                              // Don't test the one-time functions again.

      } while(PAPI_enum_cmp_event(&k,PAPI_NTV_ENUM_UMASKS,cid)==PAPI_OK);  // Get next umask entry (bits different) (should return PAPI_NOEVNT).
   } while(PAPI_enum_cmp_event(&m,PAPI_ENUM_EVENTS,cid)==PAPI_OK);         // Get next event code.

   // Round 2: Try to load all events into one Event Set.

   // Create an event set.
   ret = PAPI_create_eventset(&EventSet);                            // Try it.
   if (ret != PAPI_OK) {                                             // If that failed, report and exit.
   test_fail(__FILE__, __LINE__, "PAPI_enum_create_eventset failed.\n", ret);
   }

   m=PAPI_NATIVE_MASK;                                               // Get the PAPI NATIVE mask.
   ret=PAPI_enum_cmp_event(&m,PAPI_ENUM_FIRST,cid);                  // Begin enumeration of ALL papi counters.
   if (ret != PAPI_OK) {                                             // If that failed, report and exit.
      test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event failed.\n",
         ret);
   }

   i = 0;                                                            // To count successful adds.
   do{                                                               // Enumerate all events.
      memset(&info,0,sizeof(PAPI_event_info_t));                     // Clear event info.
      ret=PAPI_get_event_info(m,&info);                              // get name of k symbol.
      if (ret != PAPI_OK) {                                          // If that failed, report and exit.
         retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_get_event_info(%i) failed.\n", k);    // build message.
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
              continue;
         test_fail(__FILE__, __LINE__, errMsg, ret);
      }
      
      // Add it in by name.
      ret=PAPI_add_named_event(EventSet,info.symbol);                // Try to add it for counting.
      if (ret != PAPI_OK) {                                          // If that failed, report it.
         retlen = snprintf(errMsg, PAPI_MAX_STR_LEN, "PAPI_add_named_event('%s') failed.\n", info.symbol);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen)
            continue;
         test_fail( __FILE__, __LINE__, errMsg, ret);
      } else {
         i++;                                                        // success.
      }
   } while(PAPI_enum_cmp_event(&m,PAPI_ENUM_EVENTS,cid)==PAPI_OK);   // Get next event code.


   if (i != count) {                                                 // If we failed to add them all,
      sprintf(errMsg, "Test should have been able to add all %i events; failed after %i.\n", count, i);    // build message.
      test_fail(__FILE__, __LINE__, errMsg, 0);
   }      
      
   ret = PAPI_cleanup_eventset(EventSet);                            // Try a cleanup.
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset(EventSet) failed.\n", ret);
   }

   ret = PAPI_destroy_eventset(&EventSet);                           // Deallocate. No memory leaks!
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset(EventSet) failed.\n", ret);
   }

   if (!quiet) fprintf(stderr, "PCP discovered %i events; added "    // Say what we learned.
               "%i.\n", count, i);                                   // .. 
   PAPI_shutdown();                                                  // get out of papi.
	if (!quiet) fprintf(stderr, "Shutdown completed.\n");             // If we are verbose, 
	test_pass( __FILE__ );                                            // Note the test passed. 
   return 0;                                                         // Exit with all okay.
} // END main.
