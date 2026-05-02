/** file papi_xml_event_info.c
 *     @page papi_xml_event_info
 * @brief papi_xml_event_info utility.
 *     @section  NAME
 *             papi_xml_event_info - provides detailed information for PAPI events in XML format
 *
 *     @section Synopsis
 *
 *     @section Description
 *             papi_native_avail is a PAPI utility program that reports information 
 *             about the events available on the current platform in an XML format.
 *
 *             It will attempt to create an EventSet with each event in it, which
 *             can be slow.
 *
 *     @section Options
 *      <ul>
 *          <li>-h   print help message
 *          <li>-p   print only preset events
 *          <li>-n   print only native events
 *          <li>-c COMPONENT  print only events from component number COMPONENT
 * event1, event2, ...  Print only events that can be created in the same
 *      event set with the events event1, event2, etc.
 *       </ul>
 *
 *     @section Bugs
 *             There are no known bugs in this utility.
 *             If you find a bug, it should be reported to the
 *             PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"

static int EventSet;
static int preset = 1;
static int native = 1;
static int cidx = -1;

/**********************************************************************/
/* Take a string and print a version with properly escaped XML        */
/**********************************************************************/
static int
xmlize( const char *msg, FILE *f )
{
	const char *op;

	if ( !msg )
		return PAPI_OK;

	for ( op = msg; *op != '\0'; op++ ) {
		switch ( *op ) {
		case '"':
			fprintf( f, "&quot;" );
			break;
		case '&':
			fprintf( f, "&amp;" );
			break;
		case '\'':
			fprintf( f, "&apos;" );
			break;
		case '<':
			fprintf( f, "&lt;" );
			break;
		case '>':
			fprintf( f, "&gt;" );
			break;
		default:
		        fprintf( f, "%c", *op);
		}
	}

	return PAPI_OK;
}

/*************************************/
/* print hardware info in XML format */
/*************************************/
static int
papi_xml_hwinfo( FILE * f )
{
	const PAPI_hw_info_t *hwinfo;

	if ( ( hwinfo = PAPI_get_hardware_info(  ) ) == NULL )
		return PAPI_ESYS;

	fprintf( f, "<hardware>\n" );

	fprintf( f, "  <vendor string=\"");
	   xmlize( hwinfo->vendor_string, f );
	   fprintf( f,"\"/>\n");
	fprintf( f, "  <vendorCode value=\"%d\"/>\n", hwinfo->vendor );
	fprintf( f, "  <model string=\"");
	xmlize( hwinfo->model_string, f );
	   fprintf( f, "\"/>\n");
	fprintf( f, "  <modelCode value=\"%d\"/>\n", hwinfo->model );
	fprintf( f, "  <cpuRevision value=\"%f\"/>\n", hwinfo->revision );
	fprintf( f, "  <cpuID>\n" );
	fprintf( f, "    <family value=\"%d\"/>\n", hwinfo->cpuid_family );
	fprintf( f, "    <model value=\"%d\"/>\n", hwinfo->cpuid_model );
	fprintf( f, "    <stepping value=\"%d\"/>\n", hwinfo->cpuid_stepping );
	fprintf( f, "  </cpuID>\n" );
	fprintf( f, "  <cpuMaxMegahertz value=\"%d\"/>\n", hwinfo->cpu_max_mhz );
	fprintf( f, "  <cpuMinMegahertz value=\"%d\"/>\n", hwinfo->cpu_min_mhz );
	fprintf( f, "  <threads value=\"%d\"/>\n", hwinfo->threads );
	fprintf( f, "  <cores value=\"%d\"/>\n", hwinfo->cores );
	fprintf( f, "  <sockets value=\"%d\"/>\n", hwinfo->sockets );
	fprintf( f, "  <nodes value=\"%d\"/>\n", hwinfo->nnodes );
	fprintf( f, "  <cpuPerNode value=\"%d\"/>\n", hwinfo->ncpu );
	fprintf( f, "  <totalCPUs value=\"%d\"/>\n", hwinfo->totalcpus );
	fprintf( f, "</hardware>\n" );

	return PAPI_OK;
}



/****************************************************************/
/* Test if event can be added to an eventset                    */
/* (there might be existing events if specified on command line */
/****************************************************************/

static int
test_event( int evt )
{
	int retval;

	retval = PAPI_add_event( EventSet, evt );
	if ( retval != PAPI_OK ) {
		return retval;
	}

	if ( ( retval = PAPI_remove_event( EventSet, evt ) ) != PAPI_OK ) {
	   fprintf( stderr, "Error removing event from eventset\n" );
	   exit( 1 );
	}
	return PAPI_OK;
}

/***************************************/
/* Convert an event to XML             */
/***************************************/

static void
xmlize_event( FILE * f, PAPI_event_info_t * info, int num )
{

	if ( num >= 0 ) {
	   fprintf( f, "    <event index=\"%d\" name=\"",num);
	   xmlize( info->symbol, f );
	   fprintf( f, "\" desc=\"");
	   xmlize( info->long_descr, f );
	   fprintf( f, "\">\n");
	}
	else {
	   fprintf( f,"        <modifier name=\"");
	   xmlize( info->symbol, f );
	   fprintf( f,"\" desc=\"");
	   xmlize( info->long_descr, f );
	   fprintf( f,"\"> </modifier>\n");
	}

}


/****************************************/
/* Print all preset events              */
/****************************************/

static void
enum_preset_events( FILE * f, int cidx)
{
	int i, num;
	int retval;
	PAPI_event_info_t info;

	i = PAPI_PRESET_MASK;
	fprintf( f, "  <eventset type=\"PRESET\">\n" );
	num = -1;
	retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cidx );

	while ( retval == PAPI_OK ) {
	   num++;
	   retval = PAPI_get_event_info( i, &info );
	   if ( retval != PAPI_OK ) {
	     retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cidx );
	      continue;
	   }
	   if ( test_event( i ) == PAPI_OK ) {
	      xmlize_event( f, &info, num );
	      fprintf( f, "    </event>\n" );
	   }
	   retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cidx );
	}
	fprintf( f, "  </eventset>\n" );
}

/****************************************/
/* Print all native events              */
/****************************************/

static void
enum_native_events( FILE * f, int cidx)
{
	int i, k, num;
	int retval;
	PAPI_event_info_t info;

	i = PAPI_NATIVE_MASK;
	fprintf( f, "  <eventset type=\"NATIVE\">\n" );
	num = -1;
	retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cidx );

	while ( retval == PAPI_OK ) {

	   num++;
	   retval = PAPI_get_event_info( i, &info );
	   if ( retval != PAPI_OK ) {
	      retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cidx );
	      continue;
	   }

	   /* enumerate any umasks */
	   k = i;
	   if ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cidx ) == PAPI_OK ) {

		 /* add the event */
		 xmlize_event( f, &info, num );

		 /* add the event's unit masks */
		 do {
		    retval = PAPI_get_event_info( k, &info );
		    if ( retval == PAPI_OK ) {
		       if ( test_event( k )!=PAPI_OK ) {
			   continue;
		       }
		       xmlize_event( f, &info, -1 );
		    }
		 } while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cidx ) == PAPI_OK);
		 fprintf( f, "    </event>\n" );
	   } else {
              /* this event has no unit masks; test & write the event */
	      if ( test_event( i ) == PAPI_OK ) {
		 xmlize_event( f, &info, num );
		 fprintf( f, "    </event>\n" );
	      }
	   }
	   retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cidx );
	}
	fprintf( f, "  </eventset>\n" );
}

/****************************************/
/* Print usage information              */
/****************************************/

static void
usage( char *argv[] )
{
	fprintf( stderr, "Usage: %s [options] [[event1] event2 ...]\n", argv[0] );
	fprintf( stderr, "     options: -h     print help message\n" );
	fprintf( stderr, "              -p     print only preset events\n" );
	fprintf( stderr, "              -n     print only native events\n" );
	fprintf( stderr,"              -c n   print only events for component index n\n" );
	fprintf( stderr, "If event1, event2, etc., are specified, then only events\n");
	fprintf( stderr, "that can be run in addition to these events will be printed\n\n");
}

static void
parse_command_line (int argc, char **argv, int numc) {

  int i,retval;

     for( i = 1; i < argc; i++ ) {
	if ( argv[i][0] == '-' ) {
	   switch ( argv[i][1] ) {
	      case 'c':
      	                 /* only events for specified component */

                         /* UGH, what is this, the IOCCC? */
                         cidx = (i+1) < argc ? atoi( argv[(i++)+1] ) : -1;
			 if ( cidx < 0 || cidx >= numc ) {
			    fprintf( stderr,"Error: component index %d out of bounds (0..%d)\n",
				     cidx, numc - 1 );
			    usage( argv );
			    exit(1);
			 }
			 break;

	      case 'p':
		         /* only preset events */
			 preset = 1;
			 native = 0;
			 break;

	      case 'n':
		         /* only native events */
			 native = 1;
			 preset = 0;
			 break;

	      case 'h':
		         /* print help */
			 usage( argv );
			 exit(0);
			 break;

	      default:
			 fprintf( stderr, 
				     "Error: unknown option: %s\n", argv[i] );
			 usage( argv );
			 exit(1);
	   }
	} else {

	   /* If event names are specified, add them to the */
	   /* EventSet and test if other events can be run with them */

	   int code = -1;

	   retval = PAPI_event_name_to_code( argv[i], &code );
	   retval = PAPI_query_event( code );
	   if ( retval != PAPI_OK ) {
	      fprintf( stderr, "Error: unknown event: %s\n", argv[i] );
	      usage( argv );
	      exit(1);
	   }

	   retval = PAPI_add_event( EventSet, code );
	   if ( retval != PAPI_OK ) {
	      fprintf( stderr, 
                       "Error: event %s cannot be counted with others\n",
		       argv[i] );
	      usage( argv );
	      exit(1);
	   }
	}
     }

}


int
main( int argc, char **argv)
{
	int retval;
	const PAPI_component_info_t *comp;

	int numc = 0;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error!  PAPI_library_init\n");
		return retval;
	}

	/* report any return codes less than 0? */
	/* Why? */
#if 0
	retval = PAPI_set_debug( PAPI_VERB_ECONT );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_set_debug", retval );
	}
#endif

	/* Create EventSet to use */
	EventSet = PAPI_NULL;

	retval = PAPI_create_eventset( &EventSet  );
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error!  PAPI_create_eventset\n");
		return retval;
	}

	/* Get number of components */
	numc = PAPI_num_components(  );

	/* parse command line arguments */
        parse_command_line(argc,argv,numc);

	/* print XML header */
	fprintf( stdout, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" );
	fprintf( stdout, "<eventinfo>\n" );


	/* print hardware info */
	papi_xml_hwinfo( stdout );

	/* If a specific component specified, only print events from there */
	if ( cidx >= 0 ) {
	   comp = PAPI_get_component_info( cidx );

	   fprintf( stdout, "<component index=\"%d\" type=\"%s\" id=\"%s\">\n",
			    cidx, cidx ? "Unknown" : "CPU", comp->name );

	   if ( native )
	      enum_native_events( stdout, cidx);
	   if ( preset )
	      enum_preset_events( stdout, cidx);

	   fprintf( stdout, "</component>\n" );
	}
	else {
	   /* Otherwise, print info for all components */
	   for ( cidx = 0; cidx < numc; cidx++ ) {
	       comp = PAPI_get_component_info( cidx );

	       fprintf( stdout, "<component index=\"%d\" type=\"%s\" id=\"%s\">\n",
				cidx, cidx ? "Unknown" : "CPU", comp->name );

	       if ( native )
		  enum_native_events( stdout, cidx );
	       if ( preset )
		  enum_preset_events( stdout, cidx );

	       fprintf( stdout, "</component>\n" );

	       /* clean out eventset */
	       retval = PAPI_cleanup_eventset( EventSet );
	       if ( retval != PAPI_OK ) {
			fprintf(stderr,"Error!  PAPI_cleanup_eventset\n");
			return retval;
		}
	       retval = PAPI_destroy_eventset( &EventSet );
	       if ( retval != PAPI_OK ) {
			fprintf(stderr,"Error!  PAPI_destroy_eventset\n");
			return retval;
		}
	       EventSet = PAPI_NULL;

		retval = PAPI_create_eventset( &EventSet  );
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"Error!  PAPI_create_eventset\n");
			return retval;
		}

	       /* re-parse command line to set up any events specified */
	       parse_command_line (argc, argv, numc);
	   }
	}
	fprintf( stdout, "</eventinfo>\n" );

	return 0;
}
