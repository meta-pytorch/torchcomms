/*****************************************************************************
 * We use PAPI_state to get the counting state of an EventSet.This function  *
 * returns the state of the entire EventSet.                                 *
 *****************************************************************************/


 
#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }


int main()
{

   int retval; 
   int status = 0;
   int EventSet = PAPI_NULL;

   /****************************************************************************
   *  This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it  *
   * is likely that PAPI won't work correctly.If there is an error, retval     *
   * keeps track of the version number.                                        *
   ****************************************************************************/

   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   {
      printf("Library initialization error! \n");
	  exit(-1);
   }

   /*Creating the Eventset */
   if((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Instructions Executed to our EventSet */
   if ((retval=PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      ERROR_RETURN(retval);

   if ((retval=PAPI_state(EventSet, &status)) != PAPI_OK)
      ERROR_RETURN(retval);

   printstate(status);

   /* Start counting */
   if ((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   if (PAPI_state(EventSet, &status) != PAPI_OK)
      ERROR_RETURN(retval);
     
   printstate(status);       
 
   /* free the resources used by PAPI */
   PAPI_shutdown();

   exit(0);
}

int printstate(int status)
{
   if(status & PAPI_STOPPED)
      printf("Eventset is currently stopped or inactive \n");
   if(status & PAPI_RUNNING)
      printf("Eventset is currently running \n");
   if(status & PAPI_PAUSED)
      printf("Eventset is currently Paused \n");
   if(status & PAPI_NOT_INIT) 
      printf(" Eventset defined but not initialized \n");
   if(status & PAPI_OVERFLOWING)
      printf(" Eventset has overflowing enabled \n");
   if(status & PAPI_PROFILING)
      printf(" Eventset has profiling enabled \n"); 
   if(status & PAPI_MULTIPLEXING)
      printf(" Eventset has multiplexing enabled \n"); 
   return 0;
}
