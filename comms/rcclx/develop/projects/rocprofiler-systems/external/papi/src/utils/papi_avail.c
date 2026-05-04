// Define the papi_avail man page contents.
/**
  * file papi_avail.c
  *	@brief papi_avail utility.
  * @page papi_avail
  *	@section Name
  *	papi_avail - provides availability and detailed information for PAPI preset and user defined events.
  *
  *	@section Synopsis
  *	papi_avail [-adht] [-e event]
  *
  *	@section Description
  *	papi_avail is a PAPI utility program that reports information about the 
  *	current PAPI installation and supported preset and user defined events.
  *
  *	@section Options
  * <ul>
  *		<li>-h	Display help information about this utility.
  *		<li>-a	Display only the available PAPI events.
  *     <li>-c  Display only the available PAPI events after a check.
  *		<li>-d	Display PAPI event information in a more detailed format.
  *		<li>-e < event >	Display detailed event information for the named event. 
  *			This event can be a preset event, a user defined event, or a native event.
  *			If the event is a preset or a user defined event the output shows a list of native
  *			events the event is based on and the formula that is used to compute the events final value.\n
  *	</ul>
  *
  * Event filtering options 
  * <ul>
  *     <li>--br        Display branch related PAPI preset events
  *     <li>--cache     Display cache related PAPI preset events
  *     <li>--cnd       Display conditional PAPI preset events
  *     <li>--fp        Display Floating Point related PAPI preset events
  *     <li>--ins       Display instruction related PAPI preset events
  *     <li>--idl       Display Stalled or Idle PAPI preset events
  *     <li>--l1        Display level 1 cache related PAPI preset events
  *     <li>--l2        Display level 2 cache related PAPI preset events
  *     <li>--l3        Display level 3 cache related PAPI preset events
  *     <li>--mem       Display memory related PAPI preset events
  *     <li>--msc       Display miscellaneous PAPI preset events
  *     <li>--tlb       Display Translation Lookaside Buffer PAPI preset events
  * </ul>
  *	@section Bugs
  *	There are no known bugs in this utility.
  *	If you find a bug, it should be reported to the PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
  * <br>
  *	@see PAPI_derived_event_files
  *
  */

// Define the PAPI_derived_event_files man page contents.
/**
 *	@page PAPI_derived_event_files
 *	@brief Describes derived event definition file syntax.
 *
 *	@section main Derived Events
 *		PAPI provides the ability to define events whose value will be derived from multiple native events.  The list of native
 *		events to be used in a derived event and a formula which describes how to use them is provided in an event definition file.
 *		The PAPI team provides an event definition file which describes all of the supported PAPI preset events.  PAPI also allows
 *		a user to provide an event definition file that describes a set of user defined events which can extend the events PAPI
 *		normally supports.
 *
 *		This page documents the syntax of the commands which can appear in an event definition file.
 *
 * <br>
 *	@subsection rules General Rules:
 *	<ul>
 *		<li>Blank lines are ignored.</li>
 *		<li>Lines that begin with '#' are comments (they are also ignored).</li>
 *		<li>Names shown inside < > below represent values that must be provided by the user.</li>
 *		<li>If a user provided value contains white space, it must be protected with quotes.</li>
 *	</ul>
 *
 * <br>
 *	@subsection commands Commands:
 *		@par CPU,\<pmuName\>
 *		Specifies a PMU name which controls if the PRESET and EVENT commands that follow this line should
 *		be processed.  Multiple CPU commands can be entered without PRESET or EVENT commands between them to provide
 *		a list of PMU names to which the derived events that follow will apply.  When a PMU name provided in the list
 *		matches a PMU name known to the running system, the events which follow will be created.  If none of the PMU
 *		names provided in the list match a PMU name on the running system, the events which follow will be ignored.
 *		When a new CPU command follows either a PRESET or EVENT command, the PMU list is rebuilt.<br><br>
 *
 *		@par PRESET,\<eventName\>,\<derivedType\>,\<eventAttr\>,LDESC,\"\<longDesc\>\",SDESC,\"\<shortDesc\>\",NOTE,\"\<note\>\"
 *		Declare a PAPI preset derived event.<br><br>
 *
 *		@par EVENT,\<eventName\>,\<derivedType\>,\<eventAttr\>,LDESC,\"\<longDesc\>\",SDESC,\"\<shortDesc\>\",NOTE,\"\<note\>\"
 *		Declare a user defined derived event.<br><br>
 *
 *		@par Where:
 *		@par pmuName:
 *			The PMU which the following events should apply to.  A list of PMU names supported by your
 *			system can be obtained by running papi_component_avail on your system.<br>
 *		@par eventName:
 *			Specifies the name used to identify this derived event.  This name should be unique within the events on your system.<br>
 *		@par derivedType:
 *			Specifies the kind of derived event being defined (see 'Derived Types' below).<br>
 *		@par eventAttr:
 *			Specifies a formula and a list of base events that are used to compute the derived events value.  The syntax
 *			of this field depends on the 'derivedType' specified above (see 'Derived Types' below).<br>
 *		@par longDesc:
 *			Provides the long description of the event.<br>
 *		@par shortDesc:
 *			Provides the short description of the event.<br>
 *		@par note:
 *			Provides an event note.<br>
 *		@par baseEvent (used below):
 *			Identifies an event on which this derived event is based.  This may be a native event (possibly with event masks),
 *			an already known preset event, or an already known user event.<br>
 *
 * <br>
 *	@subsection notes Notes:
 *		The PRESET command has traditionally been used in the PAPI provided preset definition file.
 *		The EVENT command is intended to be used in user defined event definition files.  The code treats them
 *		the same so they are interchangeable and they can both be used in either event definition file.<br>
 *
 * <br>
 *	@subsection types Derived Types:
 *		This describes values allowed in the 'derivedType' field of the PRESET and EVENT commands.  It also
 *		shows the syntax of the 'eventAttr' field for each derived type supported by these commands.
 *		All of the derived events provide a list of one or more events which the derived event is based
 *		on (baseEvent).  Some derived events provide a formula that specifies how to compute the derived
 *		events value using the baseEvents in the list.  The following derived types are supported, the syntax
 *		of the 'eventAttr' parameter for each derived event type is shown in parentheses.<br><br>
 *
 *		@par NOT_DERIVED (\<baseEvent\>):
 *			This derived type defines an alias for the existing event 'baseEvent'.<br>
 *		@par DERIVED_ADD (\<baseEvent1\>,\<baseEvent2\>):
 *			This derived type defines a new event that will be the sum of two other
 *			events.  It has a value of 'baseEvent1' plus 'baseEvent2'.<br>
 *		@par DERIVED_PS (PAPI_TOT_CYC,\<baseEvent1\>):
 *			This derived type defines a new event that will report the number of 'baseEvent1' events which occurred
 *			per second.  It has a value of ((('baseEvent1' * cpu_max_mhz) * 1000000 ) / PAPI_TOT_CYC).  The user must
 *			provide PAPI_TOT_CYC as the first event of two events in the event list for this to work correctly.<br>
 *		@par DERIVED_ADD_PS (PAPI_TOT_CYC,\<baseEvent1\>,\<baseEvent2\>):
 *			This derived type defines a new event that will add together two event counters and then report the number
 *			which occurred per second.  It has a value of (((('baseEvent1' + baseEvent2) * cpu_max_mhz) * 1000000 ) / PAPI_TOT_CYC).
 *			The user must provide PAPI_TOT_CYC as the first event of three events in the event list for this to work correctly.<br>
 *		@par DERIVED_CMPD (\<baseEvent1\>,\<baseEvent2\):
 *			This derived type works much like the NOT_DERIVED type.  It is rarely used and it looks like the code just returns
 *			a single value returned from the kernel.  There is no calculation done to compute this events value.  Not sure why
 *			multiple input events seem to be needed to use this event type.<br>
 *		@par DERIVED_SUB (\<baseEvent1\>,\<baseEvent2\>):
 *			This derived type defines a new event that will be the difference between two other
 *			events.  It has a value of 'baseEvent1' minus 'baseEvent2'.<br>
 *		@par DERIVED_POSTFIX (\<pfFormula\>,\<baseEvent1\>,\<baseEvent2\>, ... ,\<baseEventn\>):
 *			This derived type defines a new event whose value is computed from several native events using
 *			a postfix (reverse polish notation) formula.  Its value is the result of processing the postfix
 *			formula.  The 'pfFormula' is of the form 'N0|N1|N2|5|*|+|-|' where the '|' acts as a token
 *			separator and the tokens N0, N1, and N2 are place holders that represent baseEvent0, baseEvent1,
 *			and baseEvent2 respectively.<br>
 *		@par DERIVED_INFIX (\<ifFormula\>,\<baseEvent1\>,\<baseEvent2\>, ... ,\<baseEventn\>):
 *			This derived type defines a new event whose value is computed from several native events using
 *			an infix (algebraic notation) formula.  Its value is the result of processing the infix
 *			formula.  The 'ifFormula' is of the form 'N0-(N1+(N2*5))' where the tokens N0, N1, and N2
 *			are place holders that represent baseEvent0, baseEvent1, and baseEvent2 respectively.<br>
 *
 * <br>
 *	@subsection example Example:
 *		In the following example, the events PAPI_SP_OPS, USER_SP_OPS, and ALIAS_SP_OPS will all measure the same events and return
 *		the same value.  They just demonstrate different ways to use the PRESET and EVENT event definition commands.<br><br>
 *
 *		<ul>
 *			<li># The following lines define pmu names that all share the following events</li>
 *			<li>CPU nhm</li>
 *			<li>CPU nhm-ex</li>
 *			<li>\# Events which should be defined for either of the above pmu types</li>
 *			<li>PRESET,PAPI_TOT_CYC,NOT_DERIVED,UNHALTED_CORE_CYCLES</li>
 *			<li>PRESET,PAPI_REF_CYC,NOT_DERIVED,UNHALTED_REFERENCE_CYCLES</li>
 *			<li>PRESET,PAPI_SP_OPS,DERIVED_POSTFIX,N0|N1|3|*|+|,FP_COMP_OPS_EXE:SSE_SINGLE_PRECISION,FP_COMP_OPS_EXE:SSE_FP_PACKED,NOTE,"Using a postfix formula"</li>
 *			<li>EVENT,USER_SP_OPS,DERIVED_INFIX,N0+(N1*3),FP_COMP_OPS_EXE:SSE_SINGLE_PRECISION,FP_COMP_OPS_EXE:SSE_FP_PACKED,NOTE,"Using the same formula in infix format"</li>
 *			<li>EVENT,ALIAS_SP_OPS,NOT_DERIVED,PAPI_SP_OPS,LDESC,"Alias for preset event PAPI_SP_OPS"</li>
 *			<li># End of event definitions for above pmu names and start of a section for a new pmu name.</li>
 *			<li>CPU snb</li>
 *		</ul>
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "print_header.h"

static char *
is_derived( PAPI_event_info_t * info )
{
	if ( strlen( info->derived ) == 0 )
		return ( "No" );
	else if ( strcmp( info->derived, "NOT_DERIVED" ) == 0 )
		return ( "No" );
	else if ( strcmp( info->derived, "DERIVED_CMPD" ) == 0 )
		return ( "No" );
	else
		return ( "Yes" );
}

static void
print_help( char **argv )
{
        printf( "This is the PAPI avail program.\n" );
        printf( "It provides availability and details about PAPI Presets and User-defined Events.\n" );
	printf( "PAPI Preset Event filters can be combined in a logical OR.\n" );
	printf( "Usage: %s [options]\n", argv[0] );
	printf( "Options:\n\n" );
	printf( "General command options:\n" );
	printf( "\t-h, --help       Print this help message\n" );
	printf( "\t-a, --avail      Display only available PAPI preset and user defined events\n" );
	printf( "\t-c, --check      Display only available PAPI preset and user defined events after an availability check\n" );
	printf( "\t-d, --detail     Display detailed information about events\n" );
	printf( "\t-e EVENTNAME     Display detail information about specified event\n" );
	printf( "\nEvent filtering options:\n" );
	printf( "\t--br             Display branch related PAPI preset events\n" );
	printf( "\t--cache          Display cache related PAPI preset events\n" );
	printf( "\t--cnd            Display conditional PAPI preset events\n" );
	printf( "\t--fp             Display Floating Point related PAPI preset events\n" );
	printf( "\t--ins            Display instruction related PAPI preset events\n" );
	printf( "\t--idl            Display Stalled or Idle PAPI preset events\n" );
	printf( "\t--l1             Display level 1 cache related PAPI preset events\n" );
	printf( "\t--l2             Display level 2 cache related PAPI preset events\n" );
	printf( "\t--l3             Display level 3 cache related PAPI preset events\n" );
	printf( "\t--mem            Display memory related PAPI preset events\n" );
	printf( "\t--msc            Display miscellaneous PAPI preset events\n" );
	printf( "\t--tlb            Display Translation Lookaside Buffer PAPI preset events\n" );
	printf( "\n" );
}

static int
parse_unit_masks( PAPI_event_info_t * info )
{
	char *pmask;

	if ( ( pmask = strchr( info->symbol, ':' ) ) == NULL ) {
		return ( 0 );
	}
	memmove( info->symbol, pmask, ( strlen( pmask ) + 1 ) * sizeof ( char ) );
	pmask = strchr( info->long_descr, ':' );
	if ( pmask == NULL )
		info->long_descr[0] = 0;
	else
		memmove( info->long_descr, pmask + sizeof ( char ),
				 ( strlen( pmask ) + 1 ) * sizeof ( char ) );
	return 1;
}

static int
checkCounter (int eventcode)
{
	int EventSet = PAPI_NULL;
	if (PAPI_create_eventset(&EventSet) != PAPI_OK)
		return 0;
	if (PAPI_add_event (EventSet, eventcode) != PAPI_OK)
		return 0;
	if (PAPI_cleanup_eventset (EventSet) != PAPI_OK)
		return 0;
	if (PAPI_destroy_eventset (&EventSet) != PAPI_OK)
		return 0;
	return 1;
}


/*
  Checks whether a preset event is available. If it is available,
  the function returns 1, or 0 otherwise. */

int is_preset_event_available(char *name) {

  int  event_code = 0 | PAPI_PRESET_MASK;
  PAPI_event_info_t info;
  int check_counter = 1;


  if (PAPI_enum_event( &event_code, PAPI_ENUM_FIRST ) != PAPI_OK) {
    printf("error!");
    exit(1);
  }

  /* Iterate over all the available preset events and compare them by names. */
  do {
    if ( PAPI_get_event_info( event_code, &info ) == PAPI_OK ) {
      
      if ( info.count ) {
	if ( (check_counter && checkCounter (event_code)) || !check_counter) {
	  if (strcmp(info.symbol, name) == 0)
	    return 1;
	}
      }
    }
  } while (PAPI_enum_event( &event_code, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK);

  return 0;
}



int
main( int argc, char **argv )
{
   int args, i, j, k;
   int retval;
   unsigned int filter = 0;
   int print_event_info = 0;
   char *name = NULL;
   int print_avail_only = PAPI_ENUM_EVENTS;
   int print_tabular = 1;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;
   int tot_count = 0;
   int avail_count = 0;
   int deriv_count = 0;
   int check_counter = 0;
   int event_code;

   PAPI_event_info_t n_info;

   /* Parse command line arguments */

   for( args = 1; args < argc; args++ ) {
      if ( strstr( argv[args], "-e" ) ) {
	 print_event_info = 1;
         if( (args+1 >= argc) || ( argv[args+1] == NULL ) || ( strlen( argv[args+1] ) == 0 ) ) {
	    print_help( argv );
	    exit( 1 );
	 }
	 name = argv[args + 1];
      }
      else if ( ( !strstr( argv[args], "--") && strstr( argv[args], "-c" ) ) || strstr(argv[args], "--check") )
      {
	 print_avail_only = PAPI_PRESET_ENUM_AVAIL;
         check_counter = 1;
      }
      else if ( strstr( argv[args], "-a" ))
	 print_avail_only = PAPI_PRESET_ENUM_AVAIL;
      else if ( strstr( argv[args], "-d" ) )
	 print_tabular = 0;
      else if ( strstr( argv[args], "-h" ) ) {
	 print_help( argv );
	 exit( 1 );
      } else if ( strstr( argv[args], "--br" ) )
	 filter |= PAPI_PRESET_BIT_BR;
      else if ( strstr( argv[args], "--cache" ) )
	 filter |= PAPI_PRESET_BIT_CACH;
      else if ( strstr( argv[args], "--cnd" ) )
	 filter |= PAPI_PRESET_BIT_CND;
      else if ( strstr( argv[args], "--fp" ) )
	 filter |= PAPI_PRESET_BIT_FP;
      else if ( strstr( argv[args], "--ins" ) )
	 filter |= PAPI_PRESET_BIT_INS;
      else if ( strstr( argv[args], "--idl" ) )
	 filter |= PAPI_PRESET_BIT_IDL;
      else if ( strstr( argv[args], "--l1" ) )
	 filter |= PAPI_PRESET_BIT_L1;
      else if ( strstr( argv[args], "--l2" ) )
	 filter |= PAPI_PRESET_BIT_L2;
      else if ( strstr( argv[args], "--l3" ) )
	 filter |= PAPI_PRESET_BIT_L3;
      else if ( strstr( argv[args], "--mem" ) )
	 filter |= PAPI_PRESET_BIT_MEM;
      else if ( strstr( argv[args], "--msc" ) )
	 filter |= PAPI_PRESET_BIT_MSC;
      else if ( strstr( argv[args], "--tlb" ) )
	 filter |= PAPI_PRESET_BIT_TLB;
   }

   if ( filter == 0 ) {
      filter = ( unsigned int ) ( -1 );
   }

   /* Init PAPI */

   retval = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retval != PAPI_VER_CURRENT ) {
	fprintf(stderr,"Error!  PAPI library mismatch!\n");
	return 1;
   }


	retval = PAPI_set_debug( PAPI_VERB_ECONT );
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error with PAPI_set debug!\n");
		return 1;
	}

      retval=papi_print_header("Available PAPI preset and user defined events plus hardware information.\n",
			       &hwinfo );
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error with PAPI_get_hardware_info!\n");
		return 1;
	}

      /* Code for info on just one event */

      if ( print_event_info ) {

	 if ( PAPI_event_name_to_code( name, &event_code ) == PAPI_OK ) {
	    if ( PAPI_get_event_info( event_code, &info ) == PAPI_OK ) {

	       if ( event_code & PAPI_PRESET_MASK ) {
		  printf( "%-30s%s\n%-30s%#-10x\n%-30s%d\n",
			  "Event name:", info.symbol, "Event Code:",
			  info.event_code, "Number of Native Events:",
			  info.count );
		  printf( "%-29s|%s|\n%-29s|%s|\n%-29s|%s|\n",
			  "Short Description:", info.short_descr,
			  "Long Description:", info.long_descr,
			  "Developer's Notes:", info.note );
		  printf( "%-29s|%s|\n%-29s|%s|\n", "Derived Type:",
			  info.derived, "Postfix Processing String:",
			  info.postfix );

		  for( j = 0; j < ( int ) info.count; j++ ) {
		     printf( " Native Code[%d]: %#x |%s|\n", j,
			     info.code[j], info.name[j] );
		     PAPI_get_event_info( (int) info.code[j], &n_info );
		     printf(" Number of Register Values: %d\n", n_info.count );
		     for( k = 0; k < ( int ) n_info.count; k++ ) {
			printf( " Register[%2d]: %#08x |%s|\n", k,
				n_info.code[k], n_info.name[k] );
		     }
		     printf( " Native Event Description: |%s|\n\n",
			     n_info.long_descr );
		  }

		  if (!is_preset_event_available(name)) {
		    printf("\nPRESET event %s is NOT available on this architecture!\n\n", name);
		  }

	       } else {	 /* must be a native event code */
		  printf( "%-30s%s\n%-30s%#-10x\n%-30s%d\n",
			  "Event name:", info.symbol, "Event Code:",
			  info.event_code, "Number of Register Values:",
			  info.count );
		  printf( "%-29s|%s|\n", "Description:", info.long_descr );
		  for ( k = 0; k < ( int ) info.count; k++ ) {
		      printf( " Register[%2d]: %#08x |%s|\n", k,
			      info.code[k], info.name[k] );
		  }

		  /* if unit masks exist but none are specified, process all */
		  if ( !strchr( name, ':' ) ) {
		     if ( 1 ) {
			if ( PAPI_enum_event( &event_code, PAPI_NTV_ENUM_UMASKS ) == PAPI_OK ) {
			   printf( "\nUnit Masks:\n" );
			   do {
			      retval = PAPI_get_event_info(event_code, &info );
			      if ( retval == PAPI_OK ) {
				 if ( parse_unit_masks( &info ) ) {
				    printf( "%-29s|%s|%s|\n",
					    " Mask Info:", info.symbol,
					    info.long_descr );
				    for ( k = 0; k < ( int ) info.count;k++ ) {
					printf( "  Register[%2d]:  %#08x  |%s|\n",
						k, info.code[k], info.name[k] );
				    }
				 }
			      }
			   } while ( PAPI_enum_event( &event_code,
					  PAPI_NTV_ENUM_UMASKS ) == PAPI_OK );
			}
		     }
		  }
	       }
	    }
	 } else {
	    printf( "Sorry, an event by the name '%s' could not be found.\n"
                    " Is it typed correctly?\n\n", name );
	 }
      } else {

	 /* Print *ALL* Events */

  for (i=0 ; i<2 ; i++) {
	// set the event code to fetch preset events the first time through loop and user events the second time through the loop
	if (i== 0) {
		event_code = 0 | PAPI_PRESET_MASK;
	} else {
		event_code = 0 | PAPI_UE_MASK;
	}

	/* For consistency, always ASK FOR the first event, if there is not one then nothing to process */
	if (PAPI_enum_event( &event_code, PAPI_ENUM_FIRST ) != PAPI_OK) {
		 continue;
	}

	// print heading to show which kind of events follow
	if (i== 0) {
		printf( "================================================================================\n" );
		printf( "  PAPI Preset Events\n" );
		printf( "================================================================================\n" );
	} else {
		printf( "\n");       // put a blank line after the presets before strarting the user events
		printf( "================================================================================\n" );
		printf( "  User Defined Events\n" );
		printf( "================================================================================\n" );
	}

	 if ( print_tabular ) {
	    printf( "    Name        Code    " );
	    if ( !print_avail_only ) {
	       printf( "Avail " );
	    }
	    printf( "Deriv Description (Note)\n" );
	 } else {
	    printf( "%-13s%-11s%-8s%-16s\n |Long Description|\n"
                    " |Developer's Notes|\n |Derived|\n |PostFix|\n"
                    " Native Code[n]: <hex> |name|\n",
		    "Symbol", "Event Code", "Count", "|Short Description|" );
	 }
	 do {
	    if ( PAPI_get_event_info( event_code, &info ) == PAPI_OK ) {
	       if ( print_tabular ) {
	      // if this is a user defined event or its a preset and matches the preset event filters, display its information
		  if ( (i==1) || (filter & info.event_type)) {
		     if ( print_avail_only ) {
		        if ( info.count ) {
                   if ( (check_counter && checkCounter (event_code)) || !check_counter)
                   {
                      printf( "%-13s%#x  %-5s%s",
                         info.symbol,
                         info.event_code,
                         is_derived( &info ), info.long_descr );
                   }
			}
		        if ( info.note[0] ) {
			   printf( " (%s)", info.note );
			}
			printf( "\n" );
		     } else {
			printf( "%-13s%#x  %-6s%-4s %s",
				info.symbol,
				info.event_code,
				( info.count ? "Yes" : "No" ),
				is_derived( &info ), info.long_descr );
			if ( info.note[0] ) {
			   printf( " (%s)", info.note );
			}
			printf( "\n" );
		     }
		     tot_count++;
		     if ( info.count ) {
	            if ((check_counter && checkCounter (event_code)) || !check_counter )
	              avail_count++;
		     }
		     if ( !strcmp( is_derived( &info ), "Yes" ) ) {
			deriv_count++;
		     }
		  }
	       } else {
		  if ( ( print_avail_only && info.count ) ||
		       ( print_avail_only == 0 ) )
	      {
	         if ((check_counter && checkCounter (event_code)) || !check_counter)
	         {
	           printf( "%s\t%#x\t%d\t|%s|\n |%s|\n"
			     " |%s|\n |%s|\n |%s|\n",
			     info.symbol, info.event_code, info.count,
			     info.short_descr, info.long_descr, info.note,
			     info.derived, info.postfix );
	           for ( j = 0; j < ( int ) info.count; j++ ) {
	              printf( " Native Code[%d]: %#x |%s|\n", j,
	              info.code[j], info.name[j] );
	           }
             }
		  }
		  tot_count++;
		  if ( info.count ) {
	         if ((check_counter && checkCounter (event_code)) || !check_counter )
		        avail_count++;
		  }
		  if ( !strcmp( is_derived( &info ), "Yes" ) ) {
		     deriv_count++;
		  }
	       }
	    }
	 } while (PAPI_enum_event( &event_code, print_avail_only ) == PAPI_OK);
  }
      }

	printf( "--------------------------------------------------------------------------------\n" );

	if ( !print_event_info ) {
		if ( print_avail_only ) {
			printf( "Of %d available events, %d ", avail_count, deriv_count );
		} else {
			printf( "Of %d possible events, %d are available, of which %d ",
				tot_count, avail_count, deriv_count );
		}
		if ( deriv_count == 1 ) {
			printf( "is derived.\n\n" );
		} else {
			printf( "are derived.\n\n" );
		}

		if (avail_count==0) {
			printf("No events detected!  Check papi_component_avail to find out why.\n");
			printf("\n");
		}
	}

	return 0;

}
