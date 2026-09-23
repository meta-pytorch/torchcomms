/*****************************************************************************
* This example shows how to use PAPI_overflow to set up an event set to      *
* begin registering overflows.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h"		/* This needs to be included every time you use PAPI */
#include <pthread.h>

#define OVER_FMT    "handler(%d ) Overflow at %p! bit=%#llx \n"
#define THRESHOLD 100000 
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

int total = 0;	/* we use total to track the amount of overflows that occurred */

/* THis is the handler called by PAPI_overflow*/
void
handler(int EventSet, void *address, long long overflow_vector, void *context)
{
   fprintf(stderr, OVER_FMT, EventSet, address, overflow_vector);
   total++;
}


int main ()
{
   int EventSet = PAPI_NULL;	
   /* must be set to null before calling PAPI_create_eventset */

   char errstring[PAPI_MAX_STR_LEN];
   long long (values[2])[2];
   int retval, i;
   double tmp = 0;
   int PAPI_event;		/* a place holder for an event preset */
   char event_name[PAPI_MAX_STR_LEN];


   /****************************************************************************
   *  This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it  *
   * is likely that PAPI won't work correctly.If there is an error, retval     *
   * keeps track of the version number.                                        *
   ****************************************************************************/

   if ((retval = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
   {
      printf("Library initialization error! \n");
      exit(1);
   }

   /* Here we create the eventset */
   if ((retval=PAPI_create_eventset (&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   PAPI_event = PAPI_TOT_INS;

   /* Here we are querying for the existence of the PAPI presets  */
   if (PAPI_query_event (PAPI_TOT_INS) != PAPI_OK)
   {
      PAPI_event = PAPI_TOT_CYC;

      if ((retval=PAPI_query_event (PAPI_TOT_INS)) != PAPI_OK)
         ERROR_RETURN(retval);

      printf ("PAPI_TOT_INS not available on this platform.");
      printf (" so subst PAPI_event with PAPI_TOT_CYC !\n\n");

   }


   /* PAPI_event_code_to_name is used to convert a PAPI preset from 
     its integer value to its string name. */
   if ((retval = PAPI_event_code_to_name (PAPI_event, event_name)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* add event to the event set */
   if ((retval = PAPI_add_event (EventSet, PAPI_event)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* register overflow and set up threshold */
   /* The threshold "THRESHOLD" was set to 100000 */
   if ((retval = PAPI_overflow (EventSet, PAPI_event, THRESHOLD, 0,
		                       handler)) != PAPI_OK)
      ERROR_RETURN(retval);

   printf ("Here are the addresses at which overflows occurred and overflow vectors \n");
  printf ("--------------------------------------------------------------\n");


  /* Start counting */

   if ( (retval=PAPI_start (EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   for (i = 0; i < 2000000; i++)
   {
      tmp = 1.01 + tmp;
      tmp++;
   }

   /* Stops the counters and reads the counter values into the values array */
   if ( (retval=PAPI_stop (EventSet, values[0])) != PAPI_OK)
      ERROR_RETURN(retval);


   printf ("The total no of overflows was %d\n", total);

   /* clear the overflow status */
   if ((retval = PAPI_overflow (EventSet, PAPI_event, 0, 0,
		                       handler)) != PAPI_OK)
      ERROR_RETURN(retval);

   /************************************************************************
    * PAPI_cleanup_eventset can only be used after the counter has been    *
    * stopped then it remove all events in the eventset                    *
    ************************************************************************/
   if ( (retval=PAPI_cleanup_eventset (EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Free all memory and data structures, EventSet must be empty. */
   if ( (retval=PAPI_destroy_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* free the resources used by PAPI */ 
   PAPI_shutdown();

   exit(0);
}
