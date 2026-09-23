/*****************************************************************************
* This is an example using the low level function PAPI_get_executable_info   *
* get the executable address space information. This function returns a      *
* pointer to a structure containing address information about the current    *
* program.                                                                   *
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h" /* This needs to be included every time you use PAPI */

int main()
{
   int i,tmp=0;   
   int retval;
   const PAPI_exe_info_t *prginfo = NULL;

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
    

   for(i=0;i<1000;i++)
      tmp=tmp+i;
    
   /* PAPI_get_executable_info returns a NULL if there is an error */    
   if ((prginfo = PAPI_get_executable_info()) == NULL)
   {
      printf("PAPI_get_executable_info error! \n");
      exit(1);
   }

  
   printf("Start text addess of user program is at %p\n",
              prginfo->address_info.text_start);
   printf("End text address of user program is at %p\n",
              prginfo->address_info.text_end);

   exit(0);
}
