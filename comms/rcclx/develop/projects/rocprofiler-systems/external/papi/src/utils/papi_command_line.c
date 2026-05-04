/* file papi_command_line.c
 * This simply tries to add the events listed on the command line one at a time
 * then starts and stops the counters and prints the results
*/

/**
  *	@page papi_command_line
  * @brief executes PAPI preset or native events from the command line.
  *
  *	@section Synopsis
  *		papi_command_line < event > < event > ...
  *
  *	@section Description
  *		papi_command_line is a PAPI utility program that adds named events from the 
  *		command line to a PAPI EventSet and does some work with that EventSet. 
  *		This serves as a handy way to see if events can be counted together, 
  *		and if they give reasonable results for known work.
  *
  *	@section Options
  * <ul>
  *		<li>-u          Display output values as unsigned integers
  *		<li>-x          Display output values as hexadecimal
  *		<li>-h          Display help information about this utility.
  *	</ul>
  *
  *	@section Bugs
  *		There are no known bugs in this utility.
  *		If you find a bug, it should be reported to the
  *		PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "do_loops.h"

static void
print_help( char **argv )
{
	printf( "Usage: %s [options] [EVENTNAMEs]\n", argv[0] );
	printf( "Options:\n\n" );
	printf( "General command options:\n" );
	printf( "\t-u          Display output values as unsigned integers\n" );
	printf( "\t-x          Display output values as hexadecimal\n" );
	printf( "\t-h          Print this help message\n" );
	printf( "\tEVENTNAMEs  Specify one or more preset or native events\n" );
	printf( "\n" );
	printf( "This utility performs work while measuring the specified events.\n" );
	printf( "It can be useful for sanity checks on given events and sets of events.\n" );
}


int
main( int argc, char **argv )
{
	int retval;
	int num_events;
	long long *values;
	char *success;
	PAPI_event_info_t info;
	int EventSet = PAPI_NULL;
	int i, j, event, data_type = PAPI_DATATYPE_INT64;
	int u_format = 0;
	int hex_format = 0;

	printf( "\nThis utility lets you add events from the command line "
		"interface to see if they work.\n\n" );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error! PAPI_library_init\n");
		exit(retval );
	}

	retval = PAPI_create_eventset( &EventSet );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"Error! PAPI_create_eventset\n");
		exit(retval );
	}

	values =
		( long long * ) malloc( sizeof ( long long ) * ( size_t ) argc );
	success = ( char * ) malloc( ( size_t ) argc );

	if ( success == NULL || values == NULL ) {
		fprintf(stderr,"Error allocating memory!\n");
		exit(1);
	}

	for ( num_events = 0, i = 1; i < argc; i++ ) {
		if ( !strcmp( argv[i], "-h" ) ) {
			print_help( argv );
			exit( 1 );
		} else if ( !strcmp( argv[i], "-u" ) ) {
			u_format = 1;
		} else if ( !strcmp( argv[i], "-x" ) ) {
			hex_format = 1;
		} else {
			if ( ( retval = PAPI_add_named_event( EventSet, argv[i] ) ) != PAPI_OK ) {
				printf( "Failed adding: %s\nbecause: %s\n", argv[i], 
					PAPI_strerror(retval));
			} else {
				success[num_events++] = i;
				printf( "Successfully added: %s\n", argv[i] );
			}
		}
	}

	/* Automatically pass if no events, for run_tests.sh */
	if ( num_events == 0 ) {
      free(values);
      free(success);
		printf("No events specified!\n");
		printf("Try running something like: %s PAPI_TOT_CYC\n\n",
			argv[0]);
		return 0;
	}


	printf( "\n" );

	do_flops( 1 );
	do_flush(  );

	retval = PAPI_start( EventSet );
	if (retval != PAPI_OK ) {
      free(values);
      free(success);
		fprintf(stderr,"Error! PAPI_start\n");
		exit( retval );
	}

	do_flops( NUM_FLOPS );
	do_misses( 1, L1_MISS_BUFFER_SIZE_INTS );

	retval = PAPI_stop( EventSet, values );
	if (retval != PAPI_OK ) {
      free(values);
      free(success);
		fprintf(stderr,"Error! PAPI_stop\n");
		exit( retval );
	}

	for ( j = 0; j < num_events; j++ ) {
		i = success[j];
		if (! (u_format || hex_format) ) {
			retval = PAPI_event_name_to_code( argv[i], &event );
			if (retval == PAPI_OK) {
				retval = PAPI_get_event_info(event, &info);
				if (retval == PAPI_OK) data_type = info.data_type;
				else data_type = PAPI_DATATYPE_INT64;
			}
			switch (data_type) {
			  case PAPI_DATATYPE_UINT64:
				printf( "%s : \t%llu(u)", argv[i], (unsigned long long)values[j] );
				break;
			  case PAPI_DATATYPE_FP64:
				printf( "%s : \t%0.3f", argv[i], *((double *)(&values[j])) );
				break;
			  case PAPI_DATATYPE_BIT64:
				printf( "%s : \t%#llX", argv[i], values[j] );
				break;
			  case PAPI_DATATYPE_INT64:
			  default:
				printf( "%s : \t%lld", argv[i], values[j] );
				break;
			}
			if (retval == PAPI_OK)  printf( " %s", info.units );
			printf( "\n" );
		}
		if (u_format) printf( "%s : \t%llu(u)\n", argv[i], (unsigned long long)values[j] );
		if (hex_format) printf( "%s : \t%#llX\n", argv[i], values[j] );
	}

	printf( "\n----------------------------------\n" );

   free(values);
   free(success);
	return 0;

}
