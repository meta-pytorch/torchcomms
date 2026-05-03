/******************************************************************************
 * This is an example to show how to use low level function PAPI_get_virt_cyc *
 * and PAPI_get_virt_usec.                                                    *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

int i;
double tmp;

int your_slow_code()
{

   for(i=1; i<200000; i++)
   {
      tmp= (tmp+i)/2;
   }
   return 0;
}

int main()
{
   long long s,s1, e, e1;
   int retval;


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
        
   /* Here you get initial cycles and time */
   /* No error checking is done here because this function call is always 
      successful */

   s = PAPI_get_virt_cyc();

   your_slow_code();
  
   /*Here you get final cycles and time    */
   e = PAPI_get_virt_cyc();

   s1= PAPI_get_virt_usec();

   your_slow_code();

   e1= PAPI_get_virt_usec();

   printf("Virtual cycles  : %lld\nVirtual time(ms): %lld\n",e-s,e1-s1);

   /* clean up */
   PAPI_shutdown();

   exit(0);
}

