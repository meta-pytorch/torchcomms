/*
 * This tests the use of offcore_response events
 */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "event_name_lib.h"

int main( int argc, char **argv ) {


   char *instructions_event=NULL;
   char event_name[BUFSIZ];

   char user_event[BUFSIZ];
   char kernel_event[BUFSIZ];
   char user_kernel_event[BUFSIZ];


   int retval;

   /* Default Domain */
   int EventSet_default = PAPI_NULL;
   int EventSet_default_user = PAPI_NULL;
   int EventSet_default_kernel = PAPI_NULL;
   int EventSet_default_user_kernel = PAPI_NULL;
   long long default_values[1];
   long long default_user_values[1];
   long long default_kernel_values[1];
   long long default_user_kernel_values[1];

   /* User Domain */
   int EventSet_user = PAPI_NULL;
   int EventSet_user_user = PAPI_NULL;
   int EventSet_user_kernel = PAPI_NULL;
   int EventSet_user_user_kernel = PAPI_NULL;
   long long user_values[1];
   long long user_user_values[1];
   long long user_kernel_values[1];
   long long user_user_kernel_values[1];

   /* Kernel Domain */
   int EventSet_kernel = PAPI_NULL;
   int EventSet_kernel_user = PAPI_NULL;
   int EventSet_kernel_kernel = PAPI_NULL;
   int EventSet_kernel_user_kernel = PAPI_NULL;
   long long kernel_values[1];
   long long kernel_user_values[1];
   long long kernel_kernel_values[1];
   long long kernel_user_kernel_values[1];

   /* All Domain */
   int EventSet_all = PAPI_NULL;
   int EventSet_all_user = PAPI_NULL;
   int EventSet_all_kernel = PAPI_NULL;
   int EventSet_all_user_kernel = PAPI_NULL;
   long long all_values[1];
   long long all_user_values[1];
   long long all_kernel_values[1];
   long long all_user_kernel_values[1];

   /* Two Events */
   int EventSet_two = PAPI_NULL;
   long long two_values[2];

	int quiet=0;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

   /* Init the PAPI library */
   retval = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retval != PAPI_VER_CURRENT ) {
      test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
   }


   /* Get a relevant event name */
   instructions_event=get_instructions_event(event_name, BUFSIZ);
   if (instructions_event==NULL) {
      test_skip( __FILE__, __LINE__,
                "No instructions event definition for this arch",
		 PAPI_ENOSUPP );
   }

   sprintf(user_event,"%s:u=1",instructions_event);
   sprintf(kernel_event,"%s:k=1",instructions_event);
   sprintf(user_kernel_event,"%s:u=1:k=1",instructions_event);

   /*********************************/
   /* Two Events                    */
   /*********************************/

   if (!quiet) {
      printf("\tTwo Events in same EventSet\n");
   }

   retval = PAPI_create_eventset(&EventSet_two);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_two, user_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s %s\n",user_event,PAPI_strerror(retval));
      }
	test_skip( __FILE__, __LINE__,"Could not add event",PAPI_ENOSUPP);
   }
   retval = PAPI_add_named_event(EventSet_two, kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",kernel_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_two );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_two, two_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld, %s count = %lld\n",
	    user_event,two_values[0],
	    kernel_event,two_values[1]);
   }


   /*********************************/
   /* Default Domain, Default Event */
   /*********************************/

   if (!quiet) {
      printf("\tDefault Domain\n");
   }

   retval = PAPI_create_eventset(&EventSet_default);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_default, instructions_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",instructions_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_default );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_default, default_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",instructions_event,default_values[0]);
   }


   /*********************************/
   /* Default Domain, User Event */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_default_user);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_default_user, user_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_default_user );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_default_user, default_user_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_event,default_user_values[0]);
   }

   /*********************************/
   /* Default Domain, Kernel Event */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_default_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_default_kernel, kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",kernel_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_default_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_default_kernel, default_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",kernel_event,default_kernel_values[0]);
   }

   /*****************************************/
   /* Default Domain, user and Kernel Event */
   /*****************************************/


   retval = PAPI_create_eventset(&EventSet_default_user_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_default_user_kernel, user_kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_kernel_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_default_user_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_default_user_kernel, default_user_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_kernel_event,default_user_kernel_values[0]);
   }

   /*********************************/
   /* User Domain, Default Event    */
   /*********************************/

   if (!quiet) {
      printf("\tPAPI_DOM_USER Domain\n");
   }

   retval=PAPI_set_cmp_domain(PAPI_DOM_USER, 0);

   retval = PAPI_create_eventset(&EventSet_user);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_user, instructions_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",instructions_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_user );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_user, user_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",instructions_event,user_values[0]);
   }


   /*********************************/
   /* User Domain, User Event       */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_user_user);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_user_user, user_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_user_user );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_user_user, user_user_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_event,user_user_values[0]);
   }

   /*********************************/
   /* User Domain, Kernel Event     */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_user_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_user_kernel, kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_user_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_user_kernel, user_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",kernel_event,user_kernel_values[0]);
   }

   /*****************************************/
   /* User Domain, user and Kernel Event    */
   /*****************************************/

   retval = PAPI_create_eventset(&EventSet_user_user_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_user_user_kernel, user_kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_kernel_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_user_user_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_user_user_kernel, user_user_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_kernel_event,user_user_kernel_values[0]);
   }

   /*********************************/
   /* Kernel Domain, Default Event  */
   /*********************************/

   if (!quiet) {
      printf("\tPAPI_DOM_KERNEL Domain\n");
   }

   retval=PAPI_set_cmp_domain(PAPI_DOM_KERNEL, 0);

   retval = PAPI_create_eventset(&EventSet_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_kernel, instructions_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",instructions_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_kernel, kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",instructions_event,kernel_values[0]);
   }


   /*********************************/
   /* Kernel Domain, User Event     */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_kernel_user);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_kernel_user, user_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_kernel_user );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_kernel_user, kernel_user_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_event,kernel_user_values[0]);
   }

   /*********************************/
   /* Kernel Domain, Kernel Event   */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_kernel_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_kernel_kernel, kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_kernel_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_kernel_kernel, kernel_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",kernel_event,kernel_kernel_values[0]);
   }

   /*****************************************/
   /* Kernel Domain, user and Kernel Event  */
   /*****************************************/

   retval = PAPI_create_eventset(&EventSet_kernel_user_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_kernel_user_kernel, user_kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_kernel_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_kernel_user_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_kernel_user_kernel, kernel_user_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_kernel_event,kernel_user_kernel_values[0]);
   }

   /*********************************/
   /* All Domain, Default Event  */
   /*********************************/

   if (!quiet) {
      printf("\tPAPI_DOM_ALL Domain\n");
   }

   retval=PAPI_set_cmp_domain(PAPI_DOM_ALL, 0);

   retval = PAPI_create_eventset(&EventSet_all);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_all, instructions_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",instructions_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_all );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_all, all_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",instructions_event,all_values[0]);
   }


   /*********************************/
   /* All Domain, User Event        */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_all_user);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_all_user, user_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_all_user );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_all_user, all_user_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_event,all_user_values[0]);
   }

   /*********************************/
   /* All Domain, Kernel Event      */
   /*********************************/

   retval = PAPI_create_eventset(&EventSet_all_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_all_kernel, kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_all_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_all_kernel, all_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",kernel_event,all_kernel_values[0]);
   }

   /*****************************************/
   /* All Domain, user and Kernel Event     */
   /*****************************************/

   retval = PAPI_create_eventset(&EventSet_all_user_kernel);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }


   retval = PAPI_add_named_event(EventSet_all_user_kernel, user_kernel_event);
   if (retval != PAPI_OK) {
      if ( !quiet ) {
	 fprintf(stderr,"Error trying to add %s\n",user_kernel_event);
      }
      test_fail(__FILE__, __LINE__, "adding instructions event ",retval);
   }

   retval = PAPI_start( EventSet_all_user_kernel );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet_all_user_kernel, all_user_kernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
     printf("\t\t%s count = %lld\n",user_kernel_event,all_user_kernel_values[0]);
   }

   /**************/
   /* Validation */
   /**************/

   //TODO

	test_pass( __FILE__ );

	return 0;
}
