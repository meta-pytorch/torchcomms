/* This file decodes the preset events into a csv format file */
/** file papi_decode.c
  * @brief papi_decode utility.
  *	@page papi_decode
  *	@section  NAME
  *		papi_decode - provides availability and detail information for PAPI preset events. 
  *
  *	@section Synopsis
  *		papi_decode [-ah]
  *
  *	@section Description
  *		papi_decode is a PAPI utility program that converts the PAPI presets 
  *		for the existing library into a comma separated value format that can 
  *		then be viewed or modified in spreadsheet applications or text editors,
  *		and can be supplied to PAPI_encode_events (3) as a way of adding or 
  *		modifying event definitions for specialized applications. 
  *		The format for the csv output consists of a line of field names, followed 
  *		by a blank line, followed by one line of comma separated values for each 
  *		event contained in the preset table. 
  *		A portion of this output (for Pentium 4) is shown below:
  *		@code
  *		name,derived,postfix,short_descr,long_descr,note,[native,...]
  *		PAPI_L1_ICM,NOT_DERIVED,,"L1I cache misses","Level 1 instruction cache misses",,BPU_fetch_request_TCMISS
  *		PAPI_L2_TCM,NOT_DERIVED,,"L2 cache misses","Level 2 cache misses",,BSQ_cache_reference_RD_2ndL_MISS_WR_2ndL_MISS
  *		PAPI_TLB_DM,NOT_DERIVED,,"Data TLB misses","Data translation lookaside buffer misses",,page_walk_type_DTMISS
  * @endcode
  *
  *	@section Options
  *	<ul>
  *		<li>-a	Convert only the available PAPI preset events.
  *		<li>-h	Display help information about this utility.
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

static void
print_help( void )
{
	printf( "This is the PAPI decode utility program.\n" );
	printf( "It decodes PAPI preset events into csv formatted text.\n" );
	printf( "By default all presets are decoded.\n" );
	printf( "The text goes to stdout, but can be piped to a file.\n" );
	printf( "Such a file can be edited in a text editor or spreadsheet.\n" );
	printf( "It can also be parsed by PAPI_encode_events.\n" );
	printf( "Usage:\n\n" );
	printf( "    decode [options]\n\n" );
	printf( "Options:\n\n" );
	printf( "  -a            decode only available PAPI preset events\n" );
	printf( "  -h            print this help message\n" );
	printf( "\n" );
}

int
main( int argc, char **argv )
{
	int i, j;
	int retval;
	int print_avail_only = 0;
	PAPI_event_info_t info;

	(void)argc;
	(void)argv;

	for ( i = 1; i < argc; i++ )
		if ( argv[i] ) {
			if ( !strcmp( argv[i], "-a" ) )
				print_avail_only = PAPI_PRESET_ENUM_AVAIL;
			else if ( !strcmp( argv[i], "-h" ) ) {
				print_help(  );
				exit( 1 );
			} else {
				print_help(  );
				exit( 1 );
			}
		}

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error with PAPI_library_init!\n");
		return retval;
	}

	retval = PAPI_set_debug( PAPI_VERB_ECONT );
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error with PAPI_set_debug\n");
		return retval;
	}

	i = PAPI_PRESET_MASK;
	printf
		( "name,derived,postfix,short_descr,long_descr,note,[native,...]\n\n" );

	do {
		if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
			printf( "%s,%s,%s,", info.symbol, info.derived, info.postfix );
			if ( info.short_descr[0] ) {
				printf( "\"%s\",", info.short_descr );
			} else {
				printf( "," );
			}
			if ( info.long_descr[0] ) {
				printf( "\"%s\",", info.long_descr );
			} else {
				printf( "," );
			}
			if ( info.note[0] )
				printf( "\"%s\"", info.note );

			for ( j = 0; j < ( int ) info.count; j++ )
				printf( ",%s", info.name[j] );
			printf( "\n" );
		}
	} while ( PAPI_enum_event( &i, print_avail_only ) == PAPI_OK );

	return 0;
}
