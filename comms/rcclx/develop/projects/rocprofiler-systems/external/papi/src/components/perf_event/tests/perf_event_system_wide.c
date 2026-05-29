/*
 * This tests the measuring of events using a system-wide granularity
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#ifndef __USE_GNU
#define __USE_GNU
#endif

/* For sched_setaffinity() */
#include <sched.h>


int main( int argc, char **argv ) {

	int retval;
	int EventSetDefault = PAPI_NULL;
	int EventSetUser = PAPI_NULL;
	int EventSetKernel = PAPI_NULL;
	int EventSetUserKernel = PAPI_NULL;
	int EventSetAll = PAPI_NULL;
	int EventSet4 = PAPI_NULL;
	int EventSet5 = PAPI_NULL;
	int EventSet6 = PAPI_NULL;
	int EventSet7 = PAPI_NULL;
	int EventSet8 = PAPI_NULL;
	int EventSet9 = PAPI_NULL;
	int EventSet10 = PAPI_NULL;

	int quiet=0;

	PAPI_domain_option_t domain_opt;
	PAPI_granularity_option_t gran_opt;
	PAPI_cpu_option_t cpu_opt;
	cpu_set_t mask;

	long long dom_default_values[1],
		dom_user_values[1],
		dom_kernel_values[1],
		dom_userkernel_values[1],
		dom_all_values[1];
	long long grn_thr_values[1],grn_proc_values[1];
	long long grn_sys_values[1],grn_sys_cpu_values[1];
	long long total_values[1],total_affinity_values[1];
	long long total_all_values[1];

	dom_user_values[0]=0;
	dom_userkernel_values[0]=0;
	dom_all_values[0]=0;
	grn_thr_values[0]=0;
	grn_proc_values[0]=0;
	grn_sys_values[0]=0;
	grn_sys_cpu_values[0]=0;
	total_values[0]=0;
	total_affinity_values[0]=0;
	total_all_values[0]=0;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf("\nTrying PAPI_TOT_INS with different domains:\n");
	}

	/***************************/
	/***************************/
	/* Default		   */
	/***************************/
	/***************************/

	retval = PAPI_create_eventset(&EventSetDefault);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	retval = PAPI_add_named_event(EventSetDefault, "PAPI_TOT_INS");
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Error trying to add PAPI_TOT_INS\n");
		}
		test_skip(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
	}

	if (!quiet) {
		printf("\tDefault:\t\t\t");
	}

	retval = PAPI_start( EventSetDefault );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSetDefault, dom_default_values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("%lld\n",dom_default_values[0]);
	}

	/***************************/
	/***************************/
	/* user events             */
	/***************************/
	/***************************/

	retval = PAPI_create_eventset(&EventSetUser);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	retval = PAPI_assign_eventset_component(EventSetUser, 0);

	/* we need to set domain to be as inclusive as possible */
	domain_opt.def_cidx=0;
	domain_opt.eventset=EventSetUser;
	domain_opt.domain=PAPI_DOM_USER;

	retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
	if (retval != PAPI_OK) {
		if (retval==PAPI_EPERM) {
         		test_skip( __FILE__, __LINE__,
		    		"this test; trying to set PAPI_DOM_ALL; need to run as root",
				retval);
		}
		else {
			test_fail(__FILE__, __LINE__, "setting PAPI_DOM_KERNEL",retval);
		}
	}

	retval = PAPI_add_named_event(EventSetUser, "PAPI_TOT_INS");
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Error trying to add PAPI_TOT_INS\n");
		}
		test_skip(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
	}

	if (!quiet) {
		printf("\tPAPI_DOM_USER:\t\t\t");
	}

	retval = PAPI_start( EventSetUser );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSetUser, dom_user_values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("%lld\n",dom_user_values[0]);
	}


	/***************************/
	/***************************/
	/* kernel events           */
	/***************************/
	/***************************/

	retval = PAPI_create_eventset(&EventSetKernel);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	retval = PAPI_assign_eventset_component(EventSetKernel, 0);

	/* we need to set domain to be as inclusive as possible */
	domain_opt.def_cidx=0;
	domain_opt.eventset=EventSetKernel;
	domain_opt.domain=PAPI_DOM_KERNEL;

	retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
	if (retval != PAPI_OK) {
		if (retval==PAPI_EPERM) {
         		test_skip( __FILE__, __LINE__,
		    		"this test; trying to set PAPI_DOM_ALL; need to run as root",
				retval);
		}
		else {
			test_fail(__FILE__, __LINE__, "setting PAPI_DOM_KERNEL",retval);
		}
	}

	retval = PAPI_add_named_event(EventSetKernel, "PAPI_TOT_INS");
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Error trying to add PAPI_TOT_INS\n");
		}
		test_skip(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
	}

	if (!quiet) {
		printf("\tPAPI_DOM_KERNEL:\t\t");
	}

	retval = PAPI_start( EventSetKernel );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSetKernel, dom_kernel_values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("%lld\n",dom_kernel_values[0]);
	}


	/***************************/
	/***************************/
	/* User+Kernel events      */
	/***************************/
	/***************************/

	if (!quiet) {
		printf("\tPAPI_DOM_USER|PAPI_DOM_KERNEL:\t");
	}

	retval = PAPI_create_eventset(&EventSetUserKernel);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	retval = PAPI_assign_eventset_component(EventSetUserKernel, 0);

	/* we need to set domain to be as inclusive as possible */

	domain_opt.def_cidx=0;
	domain_opt.eventset=EventSetUserKernel;
	domain_opt.domain=PAPI_DOM_USER|PAPI_DOM_KERNEL;

	retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
	if (retval != PAPI_OK) {

		if (retval==PAPI_EPERM) {
			test_skip( __FILE__, __LINE__,
				"this test; trying to set PAPI_DOM_ALL; need to run as root",
				retval);
		}
		else {
			test_fail(__FILE__, __LINE__, "setting PAPI_DOM_ALL",retval);
		}
	}


	retval = PAPI_add_named_event(EventSetUserKernel, "PAPI_TOT_INS");
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Error trying to add PAPI_TOT_INS\n");
		}
		test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
	}

	retval = PAPI_start( EventSetUserKernel );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSetUserKernel, dom_userkernel_values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("%lld\n",dom_userkernel_values[0]);
	}

   /***************************/
   /***************************/
   /* DOMAIN_ALL  events      */
   /***************************/
   /***************************/

   if (!quiet) {
      printf("\tPAPI_DOM_ALL:\t\t\t");
   }

   retval = PAPI_create_eventset(&EventSetAll);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSetAll, 0);

   /* we need to set domain to be as inclusive as possible */

   domain_opt.def_cidx=0;
   domain_opt.eventset=EventSetAll;
   domain_opt.domain=PAPI_DOM_ALL;

   retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
   if (retval != PAPI_OK) {

      if (retval==PAPI_EPERM) {
         test_skip( __FILE__, __LINE__,
		    "this test; trying to set PAPI_DOM_ALL; need to run as root",
		    retval);
      }
      else {
         test_fail(__FILE__, __LINE__, "setting PAPI_DOM_ALL",retval);
      }
   }


   retval = PAPI_add_named_event(EventSetAll, "PAPI_TOT_INS");
   if (retval != PAPI_OK) {
      if ( !quiet ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_INS\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
   }

   retval = PAPI_start( EventSetAll );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSetAll, dom_all_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
      printf("%lld\n",dom_all_values[0]);
   }


   /***************************/
   /***************************/
   /* PAPI_GRN_THR  events */
   /***************************/
   /***************************/

   if ( !quiet ) {
      printf("\nTrying different granularities:\n");
   }

   if ( !quiet ) {
      printf("\tPAPI_GRN_THR:\t\t\t");
   }

   retval = PAPI_create_eventset(&EventSet4);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet4, 0);

   /* Set the granularity to individual thread */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet4;
   gran_opt.granularity=PAPI_GRN_THR;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      test_skip( __FILE__, __LINE__,
		      "this test; trying to set PAPI_GRN_THR",
		      retval);
   }


   retval = PAPI_add_named_event(EventSet4, "PAPI_TOT_INS");
   if (retval != PAPI_OK) {
      if ( !quiet ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_INS\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
   }

   retval = PAPI_start( EventSet4 );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet4, grn_thr_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
      printf("%lld\n",grn_thr_values[0]);
   }


   /***************************/
   /***************************/
   /* PAPI_GRN_PROC  events   */
   /***************************/
   /***************************/

   if ( !quiet ) {
      printf("\tPAPI_GRN_PROC:\t\t\t");
   }

   retval = PAPI_create_eventset(&EventSet5);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet5, 0);

   /* Set the granularity to process */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet5;
   gran_opt.granularity=PAPI_GRN_PROC;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!quiet) {
         printf("Unable to set PAPI_GRN_PROC\n");
      }
   }
   else {
      retval = PAPI_add_named_event(EventSet5, "PAPI_TOT_INS");
      if (retval != PAPI_OK) {
         if ( !quiet ) {
            printf("Error trying to add PAPI_TOT_INS\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
      }

      retval = PAPI_start( EventSet5 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet5, grn_proc_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !quiet ) {
         printf("%lld\n",grn_proc_values[0]);
      }
   }



   /***************************/
   /***************************/
   /* PAPI_GRN_SYS  events    */
   /***************************/
   /***************************/

   if ( !quiet ) {
      printf("\tPAPI_GRN_SYS:\t\t\t");
   }

   retval = PAPI_create_eventset(&EventSet6);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet6, 0);

   /* Set the granularity to current cpu */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet6;
   gran_opt.granularity=PAPI_GRN_SYS;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!quiet) {
         printf("Unable to set PAPI_GRN_SYS\n");
      }
   }
   else {

	retval = PAPI_add_named_event(EventSet6, "PAPI_TOT_INS");
	if (retval != PAPI_OK) {

		if (retval == PAPI_EPERM) {
			/* FIXME: read perf_event_paranoid and see */
			if (!quiet) printf("SYS granularity not allowed, probably perf_event_paranoid permissions\n");
		}
		else {
			if ( !quiet ) {
            			printf("Error adding PAPI_TOT_INS with system granularity\n");
         		}
			test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS with system granularity",retval);
		}
	} else {

         retval = PAPI_start( EventSet6 );
         if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_start", retval );
         }

         do_flops( NUM_FLOPS );

         retval = PAPI_stop( EventSet6, grn_sys_values );
         if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
         }

         if ( !quiet ) {
            printf("%lld\n",grn_sys_values[0]);
         }
      }

   }


   /****************************/
   /****************************/
   /* PAPI_GRN_SYS_CPU  events */
   /****************************/
   /****************************/

   if ( !quiet ) {
      printf("\tPAPI_GRN_SYS_CPU:\t\t");
   }

   retval = PAPI_create_eventset(&EventSet7);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet7, 0);

   /* Set the granularity to all cpus */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet7;
   gran_opt.granularity=PAPI_GRN_SYS_CPU;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!quiet) {
         printf("Unable to set PAPI_GRN_SYS_CPU\n");
      }
   }
   else {
      retval = PAPI_add_named_event(EventSet7, "PAPI_TOT_INS");
      if (retval != PAPI_OK) {
         if ( !quiet ) {
            printf("Error trying to add PAPI_TOT_INS\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
      }

      retval = PAPI_start( EventSet7 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet7, grn_sys_cpu_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !quiet ) {
         printf("%lld\n",grn_sys_cpu_values[0]);
      }
   }


   /***************************/
   /***************************/
   /* SYS and ATTACH  events  */
   /***************************/
   /***************************/

   if ( !quiet ) {
      printf("\nPAPI_GRN_SYS plus CPU attach:\n");
   }

   if ( !quiet ) {
      printf("\tGRN_SYS, DOM_USER, CPU 0 attach:\t");
   }

   retval = PAPI_create_eventset(&EventSet8);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet8, 0);

   /* Set the granularity to system-wide */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet8;
   gran_opt.granularity=PAPI_GRN_SYS;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!quiet) {
         printf("Unable to set PAPI_GRN_SYS\n");
      }
   }
   else {
      /* we need to set to a certain cpu */

      cpu_opt.eventset=EventSet8;
      cpu_opt.cpu_num=0;

      retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
      if (retval != PAPI_OK) {
	 if (retval==PAPI_EPERM) {
		if (!quiet) {
			printf("Permission error trying to CPU_ATTACH; need to run as root\n");
		}
            test_skip( __FILE__, __LINE__,
		    "this test; trying to CPU_ATTACH; need to run as root",
		    retval);
	 }

         test_fail(__FILE__, __LINE__, "PAPI_CPU_ATTACH",retval);
      }

      retval = PAPI_add_named_event(EventSet8, "PAPI_TOT_INS");
      if (retval != PAPI_OK) {
         if ( !quiet ) {
            printf("Error trying to add PAPI_TOT_INS\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
      }

      retval = PAPI_start( EventSet8 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet8, total_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !quiet ) {
         printf("%lld\n",total_values[0]);
      }
   }


   /***************************/
   /***************************/
   /* SYS and ATTACH, bind CPU  events  */
   /***************************/
   /***************************/

   if ( !quiet ) {
      printf("\tGRN_SYS, DOM_USER, CPU 0 affinity:\t");
   }

   /* Set affinity to CPU 0 */
   CPU_ZERO(&mask);
   CPU_SET(0,&mask);
   retval=sched_setaffinity(0, sizeof(mask), &mask);

   if (retval<0) {
     if (!quiet) {
        printf("Setting affinity failed: %s\n",strerror(errno));
     }
   } else {

      retval = PAPI_create_eventset(&EventSet9);
      if (retval != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
      }

      retval = PAPI_assign_eventset_component(EventSet9, 0);

      /* Set the granularity to system-wide */

      gran_opt.def_cidx=0;
      gran_opt.eventset=EventSet9;
      gran_opt.granularity=PAPI_GRN_SYS;

      retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
      if (retval != PAPI_OK) {
         if (!quiet) {
            printf("Unable to set PAPI_GRN_SYS\n");
         }
      }
      else {
         /* we need to set to a certain cpu for uncore to work */

         cpu_opt.eventset=EventSet9;
         cpu_opt.cpu_num=0;

         retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
         if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_CPU_ATTACH",retval);
         }

         retval = PAPI_add_named_event(EventSet9, "PAPI_TOT_INS");
         if (retval != PAPI_OK) {
            if ( !quiet ) {
               printf("Error trying to add PAPI_TOT_INS\n");
            }
            test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
         }

         retval = PAPI_start( EventSet9 );
         if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_start", retval );
         }

         do_flops( NUM_FLOPS );

         retval = PAPI_stop( EventSet9, total_affinity_values );
         if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
         }

         if ( !quiet ) {
            printf("%lld\n",total_affinity_values[0]);
         }
      }
   }

   /***************************/
   /***************************/
   /* SYS and ATTACH, bind CPU  events  */
   /***************************/
   /***************************/

   if ( !quiet ) {
      printf("\tGRN_SYS, DOM_ALL, CPU 0 affinity:\t");
   }



   /* Set affinity to CPU 0 */
   CPU_ZERO(&mask);
   CPU_SET(0,&mask);
   retval=sched_setaffinity(0, sizeof(mask), &mask);

   if (retval<0) {
     if (!quiet) {
        printf("Setting affinity failed: %s\n",strerror(errno));
     }
   } else {

      retval = PAPI_create_eventset(&EventSet10);
      if (retval != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
      }

      retval = PAPI_assign_eventset_component(EventSet10, 0);

      /* Set DOM_ALL */
      domain_opt.def_cidx=0;
      domain_opt.eventset=EventSet10;
      domain_opt.domain=PAPI_DOM_ALL;

      retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
      if (retval != PAPI_OK) {

         if (retval==PAPI_EPERM) {
            test_skip( __FILE__, __LINE__,
		    "this test; trying to set PAPI_DOM_ALL; need to run as root",
		    retval);
         }
         else {
            test_fail(__FILE__, __LINE__, "setting PAPI_DOM_ALL",retval);
         }
      }

      /* Set the granularity to system-wide */

      gran_opt.def_cidx=0;
      gran_opt.eventset=EventSet10;
      gran_opt.granularity=PAPI_GRN_SYS;

      retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
      if (retval != PAPI_OK) {
         if (!quiet) {
            printf("Unable to set PAPI_GRN_SYS\n");
         }
      }
      else {
         /* we need to set to a certain cpu for uncore to work */

         cpu_opt.eventset=EventSet10;
         cpu_opt.cpu_num=0;

         retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
         if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_CPU_ATTACH",retval);
         }

         retval = PAPI_add_named_event(EventSet10, "PAPI_TOT_INS");
         if (retval != PAPI_OK) {
            if ( !quiet ) {
               printf("Error trying to add PAPI_TOT_INS\n");
            }
            test_fail(__FILE__, __LINE__, "adding PAPI_TOT_INS ",retval);
         }

         retval = PAPI_start( EventSet10 );
         if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_start", retval );
         }

         do_flops( NUM_FLOPS );

         retval = PAPI_stop( EventSet10, total_all_values );
         if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
         }

         if ( !quiet ) {
            printf("%lld\n",total_all_values[0]);
         }
      }
   }

   /**************/
   /* Validation */
   /**************/

   if ( !quiet ) {
      printf("\n");
   }

   if ( !quiet ) {
      printf("Validating:\n");
      printf("\tDOM_USER|DOM_KERNEL (%lld) > DOM_USER (%lld)\n",
             dom_userkernel_values[0],dom_user_values[0]);
   }
   if (dom_user_values[0] > dom_userkernel_values[0]) {
      test_fail( __FILE__, __LINE__, "DOM_USER too high", 0 );
   }

	if ( !quiet ) {
		printf("\n");
	}

	test_pass( __FILE__ );

	return 0;
}
