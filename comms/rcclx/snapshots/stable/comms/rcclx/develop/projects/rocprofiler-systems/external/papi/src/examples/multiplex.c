/****************************************************************************
 * Multiplexing allows more counters to be used than what is supported by   *
 * the platform, thus allowing a larger number of events to be counted      *
 * simultaneously. When a microprocessor has a very limited number of       *
 * counters that can be counted simultaneously, a large application with    *
 * many hours of run time may require days of profiling in order to gather  *
 * enough information to base a performance analysis. Multiplexing overcomes*
 * this limitation by the usage of the counters over timesharing.           *
 * This is an example demonstrating how to use PAPI_set_multiplex to        *
 * convert a standard event set to a multiplexed event set.                 *
 ****************************************************************************/ 
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "papi.h"

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

#define NUM_ITERS 10000000
#define MAX_TO_ADD 6

double c = 0.11;
void do_flops(int n)
{
    int i;
    double a = 0.5;
    double b = 6.2;

    for (i=0; i < n; i++)
        c += a * b;
    return;
}

/* Tests that we can really multiplex a lot. */
int multiplex(void)
{
   int retval, i, EventSet = PAPI_NULL, j = 0;
   long long *values;
   PAPI_event_info_t pset;
   int events[MAX_TO_ADD], number;

   /* Initialize the library */
   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
   {
      printf("Library initialization error! \n");
      exit(1);
   }

   /* initialize multiplex support */
   retval = PAPI_multiplex_init();
   if (retval != PAPI_OK)
      ERROR_RETURN(retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      ERROR_RETURN(retval);

   /* convert the event set to a multiplex event set */
   retval = PAPI_set_multiplex(EventSet);
   if (retval != PAPI_OK)
      ERROR_RETURN(retval);
/*
   retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
   if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
      ERROR_RETURN(retval);
   printf("Adding %s\n", "PAPI_TOT_INS");
*/

   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) 
   {
      retval = PAPI_get_event_info(i | PAPI_PRESET_MASK, &pset);
      if (retval != PAPI_OK)
         ERROR_RETURN(retval);

      if ((pset.count) && (pset.event_code != PAPI_TOT_CYC)) 
      {
         printf("Adding %s\n", pset.symbol);

         retval = PAPI_add_event(EventSet, pset.event_code);
         if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
            ERROR_RETURN(retval);

	     if (retval == PAPI_OK) 
            printf("Added %s\n", pset.symbol);
	     else 
            printf("Could not add %s due to resource limitation.\n", 
                  pset.symbol);

         if (retval == PAPI_OK) 
         {
            if (++j >= MAX_TO_ADD)
                break;
         }
      }
   }

   values = (long long *) malloc(MAX_TO_ADD * sizeof(long long));
   if (values == NULL)
   {
      printf("Not enough memory available. \n");
      exit(1);
   }

   if ((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   do_flops(NUM_ITERS);

   retval = PAPI_stop(EventSet, values);
   if (retval != PAPI_OK)
      ERROR_RETURN(retval);

   /* get the number of events in the event set */
   number=MAX_TO_ADD;
   if ( (retval = PAPI_list_events(EventSet, events, &number)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* print the read result */
   for (i = 0; i < MAX_TO_ADD; i++) 
   {
      retval = PAPI_get_event_info(events[i], &pset);
      if (retval != PAPI_OK)
         ERROR_RETURN(retval);
      printf("Event name: %s  value: %lld \n", pset.symbol, values[i]);
   }

   retval = PAPI_cleanup_eventset(EventSet); 
   if (retval != PAPI_OK)
      ERROR_RETURN(retval);

   retval = PAPI_destroy_eventset(&EventSet);
   if (retval != PAPI_OK)
      ERROR_RETURN(retval);
   
   /* free the resources used by PAPI */
   PAPI_shutdown();

   return (0);
}

int main(int argc, char **argv)
{

   printf("Using %d iterations\n\n", NUM_ITERS);
   printf("Does PAPI_multiplex_init() handle lots of events?\n");
   multiplex();
   exit(0);
}
