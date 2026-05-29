/*****************************************************************************
 * This example shows how to use PAPI_set_domain                             * 
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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
   long long values[2];
   PAPI_option_t options;    
   int fd;
   

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

   /* Set the domain of this EventSet to counter user mode. The domain
      will be valid for all the eventset created after this function call 
      unless you call PAPI_set_domain again */ 
   if ((retval=PAPI_set_domain(PAPI_DOM_USER)) != PAPI_OK)
      ERROR_RETURN(retval);

   if ((retval=PAPI_create_eventset(&EventSet)) != PAPI_OK)
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
   /* add some system calls */
   fd = open("/dev/zero", O_RDONLY);
   if (fd == -1)
   {
         perror("open(/dev/zero)");
         exit(1);
   }
   close(fd);


   /* Stop counting */
   if((retval=PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);
        
   printf(" Total instructions: %lld   Total Cycles: %lld \n", values[0],
            values[1]);

   /* Set the domain of this EventSet to counter user and kernel modes */ 
   if ((retval=PAPI_set_domain(PAPI_DOM_ALL)) != PAPI_OK)
      ERROR_RETURN(retval);

   EventSet = PAPI_NULL;
   if ((retval=PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Instructions Executed to our EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      ERROR_RETURN(retval);

   /* Add Total Instructions Executed to our EventSet */
   if ( (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      ERROR_RETURN(retval);
   /* Start counting */
   if((retval=PAPI_start(EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);

   poorly_tuned_function();
   /* add some system calls */
   fd = open("/dev/zero", O_RDONLY);
   if (fd == -1)
   {
         perror("open(/dev/zero)");
         exit(1);
   }
   close(fd);

   /* Stop counting */
   if((retval=PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);
        
   printf(" Total instructions: %lld   Total Cycles: %lld \n", values[0],
            values[1]);

   /* clean up */
   PAPI_shutdown();

   exit(0);
}
