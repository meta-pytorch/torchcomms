/*****************************************************************************
 * PAPI_perror converts PAPI error codes to strings,it fills the string      *
 * destination with the error message corresponding to the error code.       *
 * The function copies length worth of the error description string          *
 * corresponding to code into destination. The resulting string is always    *
 * null terminated. If length is 0, then the string is printed on stderr.    *
 * PAPI_strerror does similar but it just returns the corresponding          *
 * error string from the code.                                               *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */


int main()
{

   int retval;
   int EventSet = PAPI_NULL;
   char error_str[PAPI_MAX_STR_LEN];
        
   /****************************************************************************
   *  This part initializes the library and compares the version number of the *
   * header file, to the version of the library, if these don't match then it  *
   * is likely that PAPI won't work correctly.If there is an error, retval     *
   * keeps track of the version number.                                        *
   ****************************************************************************/

   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   {
      exit(1);
   }
  
   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
   {
      fprintf(stderr, "PAPI error %d: %s\n",retval,PAPI_strerror(retval));
      exit(1);
   }     

   /* Add Total Instructions Executed to our EventSet */

   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
   {
      PAPI_perror( "PAPI_add_event" );
      exit(1);
   }

   /* Start counting */

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
   {
      PAPI_perror( "PAPI_start" );
      exit(1);
   }

  /* We are trying to start the  counter which has already been started, 
     and this will give an error which will be passed to PAPI_perror via 
     retval and the function will then display the error string on the 
     screen.
   */ 

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
   {
      PAPI_perror( "PAPI_start" );
   }

   /* The function PAPI_strerror returns the corresponding error string 
      from the error code */ 
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
   {
      printf("%s\n",PAPI_strerror(retval));
   }

   /* finish using PAPI and free all related resources 
     (this is optional, you don't have to use it 
   */
   PAPI_shutdown (); 

   exit(0);
}
