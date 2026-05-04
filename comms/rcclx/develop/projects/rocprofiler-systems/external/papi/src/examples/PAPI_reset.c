/*****************************************************************************
 * PAPI_reset - resets the hardware event counters used by an EventSet.      *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

int poorly_tuned_function()
{  
   float tmp;
   int i;

   for(i=1; i<2000; i++)
   {
      tmp=(tmp+100)/i;
   }
   return 0;  
}

int main()
{
    int EventSet = PAPI_NULL;
    /*must be initialized to PAPI_NULL before calling PAPI_create_event*/
    
    int retval;
    unsigned  int event_code=PAPI_TOT_INS;
    /* By default monitor total instructions */

    char errstring[PAPI_MAX_STR_LEN];
    long long values[1];

   /****************************************************************************
   *  This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it  *
   * is likely that PAPI won't work correctly.If there is an error, retval     *
   * keeps track of the version number.                                        *
   ****************************************************************************/
    
   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   {
      printf("Library initialization error! \n");
      exit(1);
   }

   /* Creating the eventset */   
   if ( (retval=PAPI_create_eventset(&EventSet)) != PAPI_OK)   
      ERROR_RETURN(retval);

   /* Add Total Instructions Executed to our EventSet */
   if ((retval=PAPI_add_event(EventSet, event_code)) != PAPI_OK)
      ERROR_RETURN(retval);

      /* Start counting */
   if((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   poorly_tuned_function();
 
   /* Stop counting */
   if((retval=PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);


   printf("The first time read value is %lld\n",values[0]);

   /* This zeroes out the counters on the eventset that was created */
   if((retval=PAPI_reset(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

      /* Start counting */
   if((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   poorly_tuned_function();
 
   /* Stop counting */
   if((retval=PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

   printf("The second time read value is %lld\n",values[0]);
   
   /* free the resources used by PAPI */
   PAPI_shutdown();

   exit(0);
}


