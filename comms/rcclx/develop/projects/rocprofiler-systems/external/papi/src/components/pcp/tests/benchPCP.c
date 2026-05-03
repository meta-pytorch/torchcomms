//-----------------------------------------------------------------------------
// This test does timing on reading an event; specifically a PCP event.
// The timing for the event can be read many times, and given as an average.
// The count and event to read are given on the command line in text, e.g.
// ./benchPCP 100 "perfevent.hwcounters.instructions.value:cpu0"
// will be read in a loop 100 times, the whole thing will be timed, and the
// time divided by 100 will be reported (avg time of a read). We do this to
// increase resolution; since our time is measured in microseconds (uS) by
// averaging many reads we can get greater accuracy.
//
// We also measure the time required to initiliaze PAPI (with the component),
// and report that.
//
// We will printf() both the initialization time and read time on the same line
// in CSV format.  If no arguments are given, we will printf() a header CSV
// line. Otherwise there must be exactly two arguments. Errors are printed to
// 'stderr'. This scheme allows a shell loop to produce a csv file with a
// header and many samples, to be processed to produce descriptive statistics
// separately (by spreadsheet or another program).
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
int EVENTREADS=0;

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

int main(int argc, char **argv) {                                       // args to set two events. 
   int i,ret;
   int EventSet = PAPI_NULL;
   char errMsg[1024];                                                   // space for an error message with more info.
   char *pcpName;                                                       // The pcp name.
   if (argc == 1) {                                                     // If no arguments given,
      printf("Initialize, Event Read Avg uS\n");                        // OUTPUT Header for CSV.
      return 0;                                                         // done.
   }

   if (argc != 3) {
      fprintf(stderr, "%s:%i ERROR Invalid number of arguments; must be 0 or 2.\n", __FILE__, __LINE__); // report.
      fprintf(stderr, "%s readsToAvg Event-Name\n", argv[0]);
      exit(-1);
   }

   // Get args.
   EVENTREADS = atoi(argv[1]);                                          // get event reads.
   if (EVENTREADS < 1) {
      fprintf(stderr, "%s:%i ERROR readsToAvg must be > 0.\n", __FILE__, __LINE__); // report.
      fprintf(stderr, "%s readsToAvg Event-Name\n", argv[0]); 
      exit(-1);
   }

   pcpName  = argv[2];                                                  // collect the pcp event name.

   gettimeofday(&t1, NULL);
   ret = PAPI_library_init( PAPI_VER_CURRENT );
   gettimeofday(&t2, NULL);
 
   if (ret != PAPI_VER_CURRENT) {                                       // if we failed,
      printf("ERROR PAPI library init failed.\n");                      // Show abort in file. 
      test_fail(__FILE__, __LINE__, "PAPI_library_init failed\n", ret); // report.
   }

   printf("%9.1f,", (mConvertUsec(t2)-mConvertUsec(t1)));               // OUTPUT PAPI library init time. 

// fprintf(stderr, "Benching Event Read with PAPI %d.%d.%d\n",
//    PAPI_VERSION_MAJOR( PAPI_VERSION ),
//    PAPI_VERSION_MINOR( PAPI_VERSION ),
//    PAPI_VERSION_REVISION( PAPI_VERSION ) );

   // Library is initialized.
   ret = PAPI_create_eventset(&EventSet);                               // Create an event. 
   if (ret != PAPI_OK) {                                                // If that failed, report and exit.
      fprintf(stderr, "ERROR PAPI_create_eventset failed.\n");
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed.\n", ret);
   }
  
   ret=PAPI_add_named_event(EventSet,pcpName);                       // Try to add it for counting.
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      sprintf(errMsg, "PAPI_add_named_event('%s') failed.\n", pcpName);
      printf("ERROR %s\n", errMsg);
      test_fail( __FILE__, __LINE__, errMsg, ret);
   }

   //----------------------------------------------------------------------------------------------
   // Testing, we just read each event EVENTREADS times, and record the times read.
   // We overwrite the previous result every time, we aren't interested in the values.
   // 'volatile' to avoid compiler optimizing out the loop.
   //----------------------------------------------------------------------------------------------
   long long pcpValue;

   // Begin event timing.
   ret = PAPI_start(EventSet);                                       // start counting.
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      printf("ERROR PAPI_start EventSet failed.\n");                 // Show abort in file. 
      test_fail( __FILE__, __LINE__, "PAPI_start_event(EventSet) failed.\n", ret);  // report and exit.
   }                                                                 

   gettimeofday(&t1, NULL);
   for (i=0; i<EVENTREADS; i++) {                                    // For all the PCP iterations, 
      ret = PAPI_read(EventSet, &pcpValue);                          // .. read without a stop.
      if (ret != PAPI_OK) {                                          // .. If that failed, report it.
         printf("ERROR PAPI_read EventSet failed.\n");               // Show abort in file. 
         test_fail( __FILE__, __LINE__, "PAPI_read(EventSet) failed.\n", ret);
      }                                                              
   } 
   gettimeofday(&t2, NULL);
      
   ret = PAPI_stop(EventSet, &pcpValue);                             // stop counting, get final value.
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      printf("ERROR PAPI_stop EventSet failed.\n");                  // Show abort in file. 
      test_fail( __FILE__, __LINE__, "PAPI_stop_event(PAPIEventSet, &papiValues[FINAL]) failed.\n", ret);
   }

   printf("%9.1f\n", (mConvertUsec(t2)-mConvertUsec(t1))/((double) EVENTREADS));  // compute average, finish line.

   ret = PAPI_cleanup_eventset(EventSet);                            // Try a cleanup.
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      printf("ERROR PAPI_cleanup_eventset failed.\n");               // Show abort in file. 
      test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset(EventSet) failed.\n", ret);
   }

   ret = PAPI_destroy_eventset(&EventSet);                           // Deallocate. No memory leaks!
   if (ret != PAPI_OK) {                                             // If that failed, report it.
      printf("ERROR PAPI_destroy_eventset failed.\n");               // Show abort in file. 
      test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset(EventSet) failed.\n", ret);
   }

   //----------------------------------------------------------------------------------------------
   // Done. cleanup.
   //----------------------------------------------------------------------------------------------
   PAPI_shutdown();                                                  // get out of papi.
// fprintf(stderr, "PAPI Shutdown completed.\n");                    // If we are verbose, 
// test_pass( __FILE__ );                                            // Note the test passed. 
   return 0;                                                         // Exit with all okay.
} // END main.
