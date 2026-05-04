/* This file utility reports hardware info and native event availability on either the host
 * CPU or on one of the attached MIC devices. It is based on the papi_native_avail utility,
 * but uses offloading to run either on the host CPU or on a target device. */
/** file hybrid_native_avail.c
  *	@page papi_hybrid_native_avail
  * @brief papi_hybrid_native_avail utility.
  *	@section  NAME
  *		papi_hybrid_native_avail - provides detailed information for PAPI native events. 
  *
  *	@section Synopsis
  *
  *	@section Description
  *		papi_hybrid_native_avail is a PAPI utility program that reports information
  *		about the native events available on the current platform or on an attached MIC card.
  *		A native event is an event specific to a specific hardware platform.
  *		On many platforms, a specific native event may have a number of optional settings.
  *		In such cases, the native event and the valid settings are presented,
  *		rather than every possible combination of those settings.
  *		For each native event, a name, a description, and specific bit patterns are provided.
  *
  *	@section Options
  * <ul>
  * <li>--help, -h    print this help message
  * <li>-d            display detailed information about native events
  * <li>-e EVENTNAME  display detailed information about named native event
  * <li>-i EVENTSTR   include only event names that contain EVENTSTR
  * <li>-x EVENTSTR   exclude any event names that contain EVENTSTR
  * <li>--noumasks    suppress display of Unit Mask information
  * <li>--mic < index > report events on the specified target MIC device
  * </ul>
  *
  * Processor-specific options
  * <ul>
  * <li>--darr        display events supporting Data Address Range Restriction
  * <li>--dear        display Data Event Address Register events only
  * <li>--iarr        display events supporting Instruction Address Range Restriction
  * <li>--iear        display Instruction Event Address Register events only
  * <li>--opcm        display events supporting OpCode Matching
  * <li>--nogroups    suppress display of Event grouping information
  * </ul>
  *
  *	@section Bugs
  *		There are no known bugs in this utility.
  *		If you find a bug, it should be reported to the
  *		PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
  *
  * Modified by Gabriel Marin <gmarin@icl.utk.edu> to use offloading.
  */

//#pragma offload_attribute (push,target(mic))
//#include "papi_test.h"
//#pragma offload_attribute (pop)

#include <stdlib.h>
#include <offload.h>

#define EVT_LINE 80

typedef struct command_flags
{
	int help;
	int details;
	int named;
	int include;
	int xclude;
	char *name, *istr, *xstr;
	int darr;
	int dear;
	int iarr;
	int iear;
	int opcm;
	int umask;
	int groups;
	int mic;
	int devidx;
} command_flags_t;

static void
print_help( char **argv )
{
	printf( "This is the PAPI native avail program.\n" );
	printf( "It provides availability and detail information for PAPI native events.\n" );
	printf( "Usage: %s [options]\n", argv[0] );
	printf( "\nOptions:\n" );
	printf( "   --help, -h   print this help message\n" );
	printf( "   -d           display detailed information about native events\n" );
	printf( "   -e EVENTNAME display detailed information about named native event\n" );
	printf( "   -i EVENTSTR  include only event names that contain EVENTSTR\n" );
	printf( "   -x EVENTSTR  exclude any event names that contain EVENTSTR\n" );
	printf( "   --noumasks   suppress display of Unit Mask information\n" );
	printf( "\nProcessor-specific options\n");
	printf( "  --darr        display events supporting Data Address Range Restriction\n" );
	printf( "  --dear        display Data Event Address Register events only\n" );
	printf( "  --iarr        display events supporting Instruction Address Range Restriction\n" );
	printf( "  --iear        display Instruction Event Address Register events only\n" );
        printf( "  --opcm        display events supporting OpCode Matching\n" );
	printf( "  --nogroups    suppress display of Event grouping information\n" );
        printf( "  --mic <index> display events on the specified Xeon Phi device\n" );
	printf( "\n" );
}

static int
no_str_arg( char *arg )
{
	return ( ( arg == NULL ) || ( strlen( arg ) == 0 ) || ( arg[0] == '-' ) );
}

static void
parse_args( int argc, char **argv, command_flags_t * f )
{

	int i;

	/* Look for all currently defined commands */
	memset( f, 0, sizeof ( command_flags_t ) );
	f->umask = 1;
	f->groups = 1;

	for ( i = 1; i < argc; i++ ) {
		if ( !strcmp( argv[i], "--darr" ) )
			f->darr = 1;
		else if ( !strcmp( argv[i], "--dear" ) )
			f->dear = 1;
		else if ( !strcmp( argv[i], "--iarr" ) )
			f->iarr = 1;
		else if ( !strcmp( argv[i], "--iear" ) )
			f->iear = 1;
		else if ( !strcmp( argv[i], "--opcm" ) )
			f->opcm = 1;
		else if ( !strcmp( argv[i], "--noumasks" ) )
			f->umask = 0;
		else if ( !strcmp( argv[i], "--nogroups" ) )
			f->groups = 0;
		else if ( !strcmp( argv[i], "-d" ) )
			f->details = 1;
		else if ( !strcmp( argv[i], "--mic" ) )
		{
			f->mic = 1;
			i++;
			if ( i >= argc || no_str_arg( argv[i] ) ) {
				printf( "Specify a device index for --mic\n");
				exit(1);
			}
			f->devidx = strtol(argv[i], 0, 10);
                } else if ( !strcmp( argv[i], "-e" ) ) {
			f->named = 1;
			i++;
			f->name = argv[i];
			if ( i >= argc || no_str_arg( f->name ) ) {
				printf( "Invalid argument for -e\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-i" ) ) {
			f->include = 1;
			i++;
			f->istr = argv[i];
			if ( i >= argc || no_str_arg( f->istr ) ) {
				printf( "Invalid argument for -i\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-x" ) ) {
			f->xclude = 1;
			i++;
			f->xstr = argv[i];
			if ( i >= argc || no_str_arg( f->xstr ) ) {
				printf( "Invalid argument for -x\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-h" ) || !strcmp( argv[i], "--help" ) )
			f->help = 1;
		else {
			printf( "%s is not supported\n", argv[i] );
			exit(1);
		}
	}

	/* if help requested, print and bail */
	if ( f->help ) {
		print_help( argv);
		exit( 1 );
	}
}

static void
space_pad( char *str, int spaces )
{
	while ( spaces-- > 0 )
		strcat( str, " " );
}

static void
print_event( PAPI_event_info_t * info, int offset )
{
	unsigned int i, j = 0;
	char str[EVT_LINE + EVT_LINE];

	/* indent by offset */
	if ( offset ) {
	   printf( "|     %-73s|\n", info->symbol );
	}
	else {
	   printf( "| %-77s|\n", info->symbol );
	}

	while ( j <= strlen( info->long_descr ) ) {
	   i = EVT_LINE - 12 - 2;
	   if ( i > 0 ) {
	      str[0] = 0;
	      strcat(str,"| " );
	      space_pad( str, 11 );
	      strncat( str, &info->long_descr[j], i );
	      j += i;
	      i = ( unsigned int ) strlen( str );
	      space_pad( str, EVT_LINE - ( int ) i - 1 );
	      strcat( str, "|" );
	   }
	   printf( "%s\n", str );
	}
}

static int
parse_unit_masks( PAPI_event_info_t * info )
{
  char *pmask,*ptr;

  /* handle the PAPI component-style events which have a component:::event type */
  if ((ptr=strstr(info->symbol, ":::"))) {
    ptr+=3;
  /* handle libpfm4-style events which have a pmu::event type event name */
  } else if ((ptr=strstr(info->symbol, "::"))) {
    ptr+=2;
  }
  else {
    ptr=info->symbol;
  }

	if ( ( pmask = strchr( ptr, ':' ) ) == NULL ) {
		return ( 0 );
	}
	memmove( info->symbol, pmask, ( strlen( pmask ) + 1 ) * sizeof ( char ) );
	pmask = strchr( info->long_descr, ':' );
	if ( pmask == NULL )
		info->long_descr[0] = 0;
	else
		memmove( info->long_descr, pmask + sizeof ( char ),
				 ( strlen( pmask ) + 1 ) * sizeof ( char ) );
	return ( 1 );
}


int
main( int argc, char **argv )
{
	int i, j = 0, k;
	int retval;
	PAPI_event_info_t info;
	const PAPI_hw_info_t *hwinfo = NULL;
	command_flags_t flags;
	int enum_modifier;
	int numcmp, cid;

    int num_devices = 0;
    int target_idx = 0;
    int offload_mode = 0;
    int target_ok = 0;

	/* Parse the command-line arguments */
	parse_args( argc, argv, &flags );

    if (flags.mic)
    {
       printf("Checking for Intel(R) Xeon Phi(TM) (Target CPU) devices...\n\n");

#ifdef __INTEL_OFFLOAD
       num_devices = _Offload_number_of_devices();
#endif
       printf("Number of Target devices installed: %d\n\n",num_devices);

       if (flags.devidx >= num_devices) {
          // Run in fallback-mode
          printf("Requested device index %d is not available. Specify a device between 0 and %d\n\n",
              flags.devidx, num_devices-1);
          exit(1);
       }
       else {
          offload_mode = 1;
          target_idx = flags.devidx;
          printf("PAPI will list the native events available on device mic%d\n\n", target_idx);
       }
    }

	/* Set enum modifier mask */
	if ( flags.dear )
		enum_modifier = PAPI_NTV_ENUM_DEAR;
	else if ( flags.darr )
		enum_modifier = PAPI_NTV_ENUM_DARR;
	else if ( flags.iear )
		enum_modifier = PAPI_NTV_ENUM_IEAR;
	else if ( flags.iarr )
		enum_modifier = PAPI_NTV_ENUM_IARR;
	else if ( flags.opcm )
		enum_modifier = PAPI_NTV_ENUM_OPCM;
	else
		enum_modifier = PAPI_ENUM_EVENTS;

///    #pragma offload target(mic: target_idx) if(offload_mode) in(argc, argv) inout(TESTS_QUIET)

	/* Initialize before parsing the input arguments */
#ifdef __INTEL_OFFLOAD
    __Offload_report(1);
#endif
    #pragma offload target(mic: target_idx) if(offload_mode)

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error! PAPI_library_init\n");
		exit(retval);
	}


	if ( !TESTS_QUIET ) {
#ifdef __INTEL_OFFLOAD
       __Offload_report(1);
#endif
       #pragma offload target(mic: target_idx) if(offload_mode)
	   retval = PAPI_set_debug( PAPI_VERB_ECONT );
	   if ( retval != PAPI_OK ) {
	      fprintf(stderr,"Error! PAPI_set_debug\n");
		exit(retval);
	   }
	}

#ifdef __INTEL_OFFLOAD
    __Offload_report(1);
#endif
    #pragma offload target(mic: target_idx) if(offload_mode) nocopy(hwinfo) 
    {
	   retval = papi_print_header( "Available native events and hardware information.\n", &hwinfo );
	   fflush(stdout);
    }
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error! PAPI_get_hardware_info\n");
		exit( 2 );
	}


	/* Do this code if the event name option was specified on the commandline */
	if ( flags.named ) 
	{
	   int papi_ok = 0;
	   char *ename = flags.name;
	   int elen = 0;
	   if (ename)
	      elen = strlen(ename) + 1;
#ifdef __INTEL_OFFLOAD
       __Offload_report(1);
#endif
       #pragma offload target(mic: target_idx) if(offload_mode) in(ename:length(elen)) out(i)
	   papi_ok = PAPI_event_name_to_code(ename, &i);
	   
	   if (papi_ok == PAPI_OK)
	   {
#ifdef __INTEL_OFFLOAD
          __Offload_report(1);
#endif
          #pragma offload target(mic: target_idx) if(offload_mode) out(info)
	      papi_ok = PAPI_get_event_info(i, &info);
       }
       
	   if (papi_ok == PAPI_OK)
	   {
		  printf( "%-30s%s\n",
		 	  "Event name:", info.symbol);
		  printf( "%-29s|%s|\n", "Description:", info.long_descr );

          /* if unit masks exist but none specified, process all */
		  if ( !strchr( flags.name, ':' ) ) 
		  {
#ifdef __INTEL_OFFLOAD
            __Offload_report(1);
#endif
            #pragma offload target(mic: target_idx) if(offload_mode) inout(i)
            papi_ok = PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS);
			if (papi_ok == PAPI_OK ) 
			{
			   printf( "\nUnit Masks:\n" );
			   do 
			   {
#ifdef __INTEL_OFFLOAD
                  __Offload_report(1);
#endif
                  #pragma offload target(mic: target_idx) if(offload_mode) inout(i, info)
			      retval = PAPI_get_event_info( i, &info );
			      if ( retval == PAPI_OK ) {
			 	     if ( parse_unit_masks( &info ) ) {
				        printf( "%-29s|%s|%s|\n", " Mask Info:",
					       info.symbol, info.long_descr );
				     }
			      }
#ifdef __INTEL_OFFLOAD
                  __Offload_report(1);
#endif
                  #pragma offload target(mic: target_idx) if(offload_mode) inout(i, info)
			      papi_ok = PAPI_enum_event(&i, PAPI_NTV_ENUM_UMASKS);
			   } while (papi_ok == PAPI_OK);
			}
          }
	   } else {
	     printf("Sorry, an event by the name '%s' could not be found.\n",
		    flags.name);
	     printf("Is it typed correctly?\n\n");
	     exit( 1 );
	   }
	}
    else {

	   /* Print *ALL* available events */

#ifdef __INTEL_OFFLOAD
       __Offload_report(1);
#endif
       #pragma offload target(mic: target_idx) if(offload_mode) 
	   numcmp = PAPI_num_components(  );

	   j = 0;

	   for ( cid = 0; cid < numcmp; cid++ ) {

	       PAPI_component_info_t component;
//	       if (offload_mode)  // I must allocate local memory to receive the result
//	          component = (PAPI_component_info_t*)malloc(sizeof(PAPI_component_info_t));
//           #pragma offload target(mic: target_idx) if(offload_mode) out(*component:length(sizeof(PAPI_component_info_t)) alloc_if(0) free_if(0))
#ifdef __INTEL_OFFLOAD
           __Offload_report(1);
#endif
           #pragma offload target(mic: target_idx) if(offload_mode) out(component)
           {
	          memcpy(&component, PAPI_get_component_info(cid), sizeof(PAPI_component_info_t));
	       }

	       /* Skip disabled components */
	       if (component.disabled) continue;

	       printf( "===============================================================================\n" );
	       printf( " Native Events in Component: %s\n",component.name);
	       printf( "===============================================================================\n" );

	       /* Always ASK FOR the first event */
	       /* Don't just assume it'll be the first numeric value */
	       i = 0 | PAPI_NATIVE_MASK;

#ifdef __INTEL_OFFLOAD
           __Offload_report(1);
#endif
           #pragma offload target(mic: target_idx) if(offload_mode) inout(i)
	       retval=PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );

	       do 
	       {
			  memset( &info, 0, sizeof ( info ) );
#ifdef __INTEL_OFFLOAD
              __Offload_report(1);
#endif
              #pragma offload target(mic: target_idx) if(offload_mode) inout(info)
			  retval = PAPI_get_event_info( i, &info );

			  /* This event may not exist */
			  if ( retval != PAPI_OK )
				 goto endloop;

			  /* Bail if event name doesn't contain include string */
			  if ( flags.include ) {
			  	 if ( !strstr( info.symbol, flags.istr ) ) {
				    goto endloop;
				 }
			  }

			  /* Bail if event name does contain exclude string */
			  if ( flags.xclude ) {
			  	 if ( strstr( info.symbol, flags.xstr ) )
				    goto endloop;
			  }

			  /* count only events that are actually processed */
			  j++;

			  print_event( &info, 0 );

			  if (flags.details) {
				if (info.units[0]) printf( "|     Units: %-67s|\n", 
							   info.units );
			  }

/*		modifier = PAPI_NTV_ENUM_GROUPS returns event codes with a
			groups id for each group in which this
			native event lives, in bits 16 - 23 of event code
			terminating with PAPI_ENOEVNT at the end of the list.
*/

			  /* This is an IBM Power issue */
			  if ( flags.groups ) {
			     int papi_ok = 0;
				 k = i;
#ifdef __INTEL_OFFLOAD
                 __Offload_report(1);
#endif
                 #pragma offload target(mic: target_idx) if(offload_mode) inout(k)
				 papi_ok = PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_GROUPS, cid);
				 if (papi_ok == PAPI_OK ) 
				 {
				    printf("Groups: ");
				    do {
				       printf( "%4d", ( ( k & PAPI_NTV_GROUP_AND_MASK ) >>
				              PAPI_NTV_GROUP_SHIFT ) - 1 );
#ifdef __INTEL_OFFLOAD
                       __Offload_report(1);
#endif
                       #pragma offload target(mic: target_idx) if(offload_mode) inout(k)
                       papi_ok = PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_GROUPS, cid);
				    } while (papi_ok==PAPI_OK );
				    printf( "\n" );
				 }
			  }

			  /* Print umasks */
			  /* components that don't have them can just ignore */

              if ( flags.umask ) 
              {
                 int papi_ok = 0;
				 k = i;
#ifdef __INTEL_OFFLOAD
                 __Offload_report(1);
#endif
                 #pragma offload target(mic: target_idx) if(offload_mode) inout(k)
				 papi_ok = PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_UMASKS, cid);
				 if (papi_ok == PAPI_OK ) 
				 {
					do {
#ifdef __INTEL_OFFLOAD
                       __Offload_report(1);
#endif
                       #pragma offload target(mic: target_idx) if(offload_mode) inout(info)
				       retval = PAPI_get_event_info(k, &info);
				       if ( retval == PAPI_OK ) {
				          if (parse_unit_masks( &info ))
				             print_event(&info, 2);
				       }
#ifdef __INTEL_OFFLOAD
                       __Offload_report(1);
#endif
                       #pragma offload target(mic: target_idx) if(offload_mode) inout(k)
                       papi_ok = PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_UMASKS, cid);
					} while (papi_ok == PAPI_OK);
				 }
			  }
			  printf( "--------------------------------------------------------------------------------\n" );

endloop:
#ifdef __INTEL_OFFLOAD
              __Offload_report(1);
#endif
              #pragma offload target(mic: target_idx) if(offload_mode) inout(i)
              retval=PAPI_enum_cmp_event(&i, enum_modifier, cid);
	       } while (retval == PAPI_OK );
	   }

	   printf("\n");
	   printf( "Total events reported: %d\n", j );
	}

	return 0;
}
