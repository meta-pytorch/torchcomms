/*
* File:    profile.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if (!defined(NO_DLFCN) && !defined(_BGL) && !defined(_BGP))
#include <dlfcn.h>
#endif

#include "papi.h"
#include "papi_test.h"

void print_shlib_info_map(const PAPI_shlib_info_t *shinfo, int quiet)
{
	PAPI_address_map_t *map = shinfo->map;
	int i;
	if (NULL == map) {
		test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info", 1);
	}

	if (!quiet) for ( i = 0; i < shinfo->count; i++ ) {
		printf( "Library: %s\n", map->name );
		printf( "Text start: %p, Text end: %p\n", map->text_start,
				map->text_end );
		printf( "Data start: %p, Data end: %p\n", map->data_start,
				map->data_end );
		printf( "Bss start: %p, Bss end: %p\n", map->bss_start, map->bss_end );

		if ( strlen( &(map->name[0]) ) == 0 )
			test_fail( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
		if ( ( map->text_start == 0x0 ) || ( map->text_end == 0x0 ) ||
			 ( map->text_start >= map->text_end ) )
			test_fail( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
/*
       if ((map->data_start == 0x0) || (map->data_end == 0x0) ||
	   (map->data_start >= map->data_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
       if (((map->bss_start) && (!map->bss_end)) ||
	   ((!map->bss_start) && (map->bss_end)) ||
	   (map->bss_start > map->bss_end))
	 test_fail(__FILE__, __LINE__, "PAPI_get_shared_lib_info",1);
*/

		map++;
	}
}

void display( char *msg )
{
	int i;
	for (i=0; i<64; i++)
	{
		printf( "%1d", (msg[i] ? 1 : 0) );
	}
	printf("\n");
}

int
main( int argc, char **argv )
{
	int retval,quiet;

	const PAPI_shlib_info_t *shinfo;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if ( ( shinfo = PAPI_get_shared_lib_info(  ) ) == NULL ) {
		test_skip( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
	}

	if ( ( shinfo->count == 0 ) && ( shinfo->map ) ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
	}

	print_shlib_info_map(shinfo, quiet);

	/* Needed for debugging, so you can ^Z and stop the process, */
	/* inspect /proc to see if it's right */
	sleep( 1 );

#ifndef NO_DLFCN
	{

	const char *_libname = "libcrypt.so";
	void *handle;
	void ( *setkey) (const char *key);
	void ( *encrypt) (char block[64], int edflag);
	char key[64]={
		1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
		1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
		1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
		1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
	}; /* bit pattern for key */
	char orig[64];      /* bit pattern for messages */
	char txt[64];      	    /* bit pattern for messages */

	int oldcount;

	handle = dlopen( _libname, RTLD_NOW );
	if ( !handle ) {
		printf( "dlopen: %s\n", dlerror(  ) );
		if (!quiet) printf( "Did you forget to set the environmental "
			"variable LIBPATH (in AIX) or "
			"LD_LIBRARY_PATH (in linux) ?\n" );
			test_fail( __FILE__, __LINE__, "dlopen", 1 );
	}

	setkey = dlsym( handle, "setkey" );
	encrypt = dlsym( handle, "encrypt" );
	if ( setkey == NULL || encrypt == NULL) {
		if (!quiet) printf( "dlsym: %s\n", dlerror(  ) );
		test_fail( __FILE__, __LINE__, "dlsym", 1 );
	}

	memset(orig,0,64);
	memcpy(txt,orig,64);
	setkey(key);

	if (!quiet) {
		printf("original  ");
		display(txt);
	}

	encrypt(txt, 0);   /* encode */

	if (!quiet) {
		printf("encrypted ");
		display(txt);
	}

	if (!memcmp(txt,orig,64)) {
		test_fail( __FILE__, __LINE__, "encode", 1 );
	}

	encrypt(txt, 1);   /* decode */

	if (!quiet) {
		printf("decrypted ");
		display(txt);
	}

	if (memcmp(txt,orig,64)) {
		test_fail( __FILE__, __LINE__, "decode", 1 );
	}

	oldcount = shinfo->count;

	if ( ( shinfo = PAPI_get_shared_lib_info(  ) ) == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
	}

	/* Needed for debugging, so you can ^Z and stop the process, */
	/* inspect /proc to see if it's right */
	sleep( 1 );

	if ( ( shinfo->count == 0 ) && ( shinfo->map ) ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
	}

	if ( shinfo->count <= oldcount ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_shared_lib_info", 1 );
	}

	print_shlib_info_map(shinfo, quiet);

	/* Needed for debugging, so you can ^Z and stop the process, */
	/* inspect /proc to see if it's right */
	sleep( 1 );

	dlclose( handle );

	}
#endif

	test_pass( __FILE__ );

	return 0;

}
