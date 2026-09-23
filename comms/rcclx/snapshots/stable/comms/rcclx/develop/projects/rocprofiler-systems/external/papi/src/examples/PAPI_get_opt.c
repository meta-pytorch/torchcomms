/*****************************************************************************
 * This is an example using the low level function PAPI_get_opt to query the *
 * option settings of the PAPI library or a specific eventset created by the *
 * PAPI_create_eventset function. PAPI_set_opt is used on the otherhand to   *
 * set PAPI library or event set options.                                    *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

   int num, retval, EventSet = PAPI_NULL;
   PAPI_option_t options;    
   long long values[2];

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

   /*PAPI_get_opt returns a negative number if there is an error */

   /* This call returns the maximum available hardware counters */
   if((num = PAPI_get_opt(PAPI_MAX_HWCTRS,NULL)) <= 0)
      ERROR_RETURN(num);


   printf("This machine has %d counters.\n",num);

   if ((retval=PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Set the domain of this EventSet to counter user and 
      kernel modes for this process.                      */
        
   memset(&options,0x0,sizeof(options));
   
   options.domain.eventset = EventSet;
   /* Default domain is PAPI_DOM_USER */
   options.domain.domain = PAPI_DOM_ALL;
   /* this sets the options for the domain */
   if ((retval=PAPI_set_opt(PAPI_DOMAIN, &options)) != PAPI_OK)
      ERROR_RETURN(retval);
   /* Add Total Instructions Executed event to the EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Cycles Executed event to the EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Start counting */
   if((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   poorly_tuned_function();

   /* Stop counting */
   if((retval=PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

   printf(" Total instructions: %lld   Total Cycles: %lld \n", values[0],
            values[1]);
   
   /* clean up */
   PAPI_shutdown();

   exit(0);
}
