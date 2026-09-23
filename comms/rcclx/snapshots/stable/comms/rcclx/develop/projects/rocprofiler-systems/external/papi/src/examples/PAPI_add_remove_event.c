/*****************************************************************************
* This example shows how to use PAPI_add_event, PAPI_start, PAPI_read,       *
*  PAPI_stop and PAPI_remove_event.                                          *
******************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

#define NUM_EVENTS 2
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }

int main()
{
   int EventSet = PAPI_NULL;
   int tmp, i;
   /*must be initialized to PAPI_NULL before calling PAPI_create_event*/

   long long values[NUM_EVENTS];
   /*This is where we store the values we read from the eventset */
    
   /* We use number to keep track of the number of events in the EventSet */ 
   int retval, number;
   
   char errstring[PAPI_MAX_STR_LEN];
  
   /*************************************************************************** 
   *  This part initializes the library and compares the version number of the*
   * header file, to the version of the library, if these don't match then it *
   * is likely that PAPI won't work correctly.If there is an error, retval    *
   * keeps track of the version number.                                       *
   ***************************************************************************/


   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
      ERROR_RETURN(retval);
     
     
   /* Creating the eventset */              
   if ( (retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Instructions Executed to the EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Cycles event to the EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* get the number of events in the event set */
   number = 0;
   if ( (retval = PAPI_list_events(EventSet, NULL, &number)) != PAPI_OK)
      ERROR_RETURN(retval);

   printf("There are %d events in the event set\n", number);

   /* Start counting */

   if ( (retval = PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);
   
   /* you can replace your code here */
   tmp=0;
   for (i = 0; i < 2000000; i++)
   {
      tmp = i + tmp;
   }

  
   /* read the counter values and store them in the values array */
   if ( (retval=PAPI_read(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

   printf("The total instructions executed for the first loop are %lld \n", values[0] );
   printf("The total cycles executed for the first loop are %lld \n",values[1]);
  
   /* our slow code again */
   tmp=0;
   for (i = 0; i < 2000000; i++)
   {
      tmp = i + tmp;
   }

   /* Stop counting and store the values into the array */
   if ( (retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

   printf("Total instructions executed are %lld \n", values[0] );
   printf("Total cycles executed are %lld \n",values[1]);

   /* Remove event: We are going to take the PAPI_TOT_INS from the eventset */
   if( (retval = PAPI_remove_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      ERROR_RETURN(retval);
   printf("Removing PAPI_TOT_INS from the eventset\n"); 

   /* Now we list how many events are left on the event set */
   number = 0;
   if ((retval=PAPI_list_events(EventSet, NULL, &number))!= PAPI_OK)
      ERROR_RETURN(retval);

   printf("There is only %d event left in the eventset now\n", number);

   /* free the resources used by PAPI */
   PAPI_shutdown();
 
   exit(0);
}


