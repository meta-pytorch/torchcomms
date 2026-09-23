/* This file utility reports hardware info and native event availability */
/** file papi_native_avail.c
 *	@page papi_native_avail
 * @brief papi_native_avail utility.
 *	@section  NAME
 *		papi_native_avail - provides detailed information for PAPI native events. 
 *
 *	@section Synopsis
 *
 *	@section Description
 *		papi_native_avail is a PAPI utility program that reports information 
 *		about the native events available on the current platform.
 *		A native event is an event specific to a specific hardware platform. 
 *		On many platforms, a specific native event may have a number of optional settings. 
 *		In such cases, the native event and the valid settings are presented, 
 *		rather than every possible combination of those settings.
 *		For each native event, a name, a description, and specific bit patterns are provided.
 *
 *	@section Options
 * <ul>
 * <li>--help, -h    print this help message
 * <li>--check, -c   attempts to add each event
 * <li>-sde FILE     lists SDEs that are registered by the library or executable in FILE
 * <li>-e EVENTNAME  display detailed information about named native event
 * <li>-i EVENTSTR   include only event names that contain EVENTSTR
 * <li>-x EVENTSTR   exclude any event names that contain EVENTSTR
 * <li>--noqual      suppress display of event qualifiers (mask and flag) information\n
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <dlfcn.h>

#include "papi.h"
#include "print_header.h"
#if SDE
#include "sde_lib/sde_lib.h"
#endif

#define EVT_LINE 80
#define EVT_LINE_BUF_SIZE 4096

typedef struct command_flags
{
	int help;
	int named;
	int include;
	int xclude;
	int check;
	int list_sdes;
	char *path, *name, *istr, *xstr;
	int darr;
	int dear;
	int iarr;
	int iear;
	int opcm;
	int qualifiers;
	int groups;
} command_flags_t;

static void
print_help( char **argv )
{
	printf( "This is the PAPI native avail program.\n" );
	printf( "It provides availability and details about PAPI Native Events.\n" );
	printf( "Usage: %s [options]\n", argv[0] );
        printf( "Options:\n\n" );
	printf( "\nGeneral command options:\n" );
	printf( "\t-h, --help       print this help message\n" );
	printf( "\t-c, --check      attempts to add each event\n");
	printf( "\t-sde FILE        lists SDEs that are registered by the library or executable in FILE\n" );
	printf( "\t-e EVENTNAME     display detailed information about named native event\n" );
	printf( "\t-i EVENTSTR      include only event names that contain EVENTSTR\n" );
	printf( "\t-x EVENTSTR      exclude any event names that contain EVENTSTR\n" );
	printf( "\t--noqual         suppress display of event qualifiers (mask and flag) information\n" );
	printf( "\nProcessor-specific options:\n");
	printf( "\t--darr           display events supporting Data Address Range Restriction\n" );
	printf( "\t--dear           display Data Event Address Register events only\n" );
	printf( "\t--iarr           display events supporting Instruction Address Range Restriction\n" );
	printf( "\t--iear           display Instruction Event Address Register events only\n" );
	printf( "\t--opcm           display events supporting OpCode Matching\n" );
	printf( "\t--nogroups       suppress display of Event grouping information\n" );
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
	f->qualifiers = 1;
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
		else if ( !strcmp( argv[i], "--noqual" ) )
			f->qualifiers = 0;
		else if ( !strcmp( argv[i], "--nogroups" ) )
			f->groups = 0;
		else if ( !strcmp( argv[i], "-e" ) ) {
			f->named = 1;
			i++;
			if ( i < argc )
			    f->name = argv[i];
			if ( no_str_arg( f->name ) ) {
				printf( "Invalid argument for -e\n");
				exit(1);
			}
		}
#if SDE
		else if ( !strcmp( argv[i], "-sde" ) ) {
			f->list_sdes = 1;
			i++;
			if ( i < argc )
			    f->path = argv[i];
			if ( no_str_arg( f->path ) ) {
				printf( "Invalid argument for -sde\n");
				exit(1);
			}
		}
#endif
		else if ( !strcmp( argv[i], "-i" ) ) {
			f->include = 1;
			i++;
			if ( i < argc )
			    f->istr = argv[i];
			if ( no_str_arg( f->istr ) ) {
				printf( "Invalid argument for -i\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-x" ) ) {
			f->xclude = 1;
			i++;
			if ( i < argc )
			    f->xstr = argv[i];
			if ( no_str_arg( f->xstr ) ) {
				printf( "Invalid argument for -x\n");
				exit(1);
			}
		} else if ( strstr( argv[i], "-h" ) ) {
			f->help = 1;
		} else if ( !strcmp( argv[i], "-c" ) || !strcmp( argv[i], "--check" ) ) {
			f->check = 1;
		} else {
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

unsigned int event_available = 0;
unsigned int event_output_buffer_size = 0;
char *event_output_buffer = NULL;

static void
check_event( PAPI_event_info_t * info )
{
	int EventSet = PAPI_NULL;

	// if this event has already passed the check test, no need to try this one again
	if (event_available) {
		return;
	}

	if (PAPI_create_eventset (&EventSet) == PAPI_OK) {
		if (PAPI_add_named_event (EventSet, info->symbol) == PAPI_OK) {
			PAPI_remove_named_event (EventSet, info->symbol);
			event_available = 1;
		} // else printf("********** PAPI_add_named_event( %s ) failed: event could not be added \n", info->symbol);
		if ( PAPI_destroy_eventset( &EventSet ) != PAPI_OK ) {
			printf("**********  Call to destroy eventset failed when trying to check event '%s'  **********\n", info->symbol);
		}
	}

	return;
}

static int
format_event_output( PAPI_event_info_t * info, int offset)
{
	unsigned int i, j = 0;
	char event_line_buffer[EVT_LINE_BUF_SIZE];
	char event_line_units[100];

	/* indent by offset */
	if ( offset ) {
		// this one is used for event qualifiers
		sprintf(event_line_buffer, "|     %-73s|\n", info->symbol);
	}
	else {
		// this one is used for new events
		sprintf(event_line_buffer, "| %-73s%4s|\n", info->symbol, "<-->");
	}

	while ( j <= strlen( info->long_descr ) ) {
		// The event_line_buffer is used to collect an event or mask name and its description.
		// The description will be folded to keep the length of output lines reasonable.  So this
		// buffer may contain multiple lines of print output.  Check to make sure there is room
		// for another line of print output.  If there is not enough room for another output line
		// just exit the loop and truncate the description field (the buffer is big enough this
		// should not happen).
		if ((EVT_LINE_BUF_SIZE - strlen(event_line_buffer)) < EVT_LINE) {
			printf ("Event or mask description has been truncated.\n");
			break;
		}

		// get amount of description that will fit in an output line
		i = EVT_LINE - 12 - 2;
		// start of a description line
		strcat(event_line_buffer,"|            " );
		// if we need to copy less than what fits in this line, move it and exit loop
		if (i > strlen(&info->long_descr[j])) {
			strcat( event_line_buffer, &info->long_descr[j]);
			space_pad( event_line_buffer, i - strlen(&info->long_descr[j]));
			strcat( event_line_buffer, "|\n" );
			break;
		}

		// move what will fit into the line then loop back to do the rest in a new line
		int k = strlen(event_line_buffer);
		strncat( event_line_buffer, &info->long_descr[j], i );
		event_line_buffer[k+i] = '\0';
		strcat( event_line_buffer, "|\n" );

		// bump index past what we copied
		j += i;
	}

	// also show the units for this event if a unit name has been set
	event_line_units[0] = '\0';
	if (info->units[0] != 0) {
		sprintf(event_line_units, "|     Units: %-66s|\n", info->units );
	}

	// get the amount of used space in the output buffer
	int out_buf_used = 0;
	if ((event_output_buffer_size > 0) && (event_output_buffer != NULL)) {
		out_buf_used = strlen(event_output_buffer);
	}

	// if this will not fit in output buffer, make it bigger
	if (event_output_buffer_size < out_buf_used + strlen(event_line_buffer) + strlen(event_line_units) + 1) {
		if (event_output_buffer_size == 0) {
			event_output_buffer_size = 1024;
			event_output_buffer = calloc(1, event_output_buffer_size);
		} else {
			event_output_buffer_size += 1024;
			event_output_buffer = realloc(event_output_buffer, event_output_buffer_size);
		}
	}

	// make sure we got the memory we asked for
	if (event_output_buffer == NULL) {
		fprintf(stderr,"Error!  Allocation of output buffer memory failed.\n");
		return errno;
	}

	strcat(event_output_buffer, event_line_buffer);
	strcat(event_output_buffer, event_line_units);

	return 0;
}

static void
print_event_output(int val_flag)
{
	// first we need to update the available flag at the beginning of the buffer
	// this needs to reflect if this event name by itself or the event name with one of the qualifiers worked
	// if none of the combinations worked then we will show the event as not available
	char *val_flag_ptr = strstr(event_output_buffer, "<-->");
	if (val_flag_ptr != NULL) {
		if ((val_flag) && (event_available == 0)) {
			// event is not available, update the place holder (replace the <--> with <NA>)
			*(val_flag_ptr+1) = 'N';
			*(val_flag_ptr+2) = 'A';
		} else {
			event_available = 0;       // reset this flag for next event
			// event is available, just remove the place holder (replace the <--> with spaces)
			*val_flag_ptr = ' ';
			*(val_flag_ptr+1) = ' ';
			*(val_flag_ptr+2) = ' ';
			*(val_flag_ptr+3) = ' ';
		}
	}

	// now we can finally send this events output to the user
	printf( "%s", event_output_buffer);
//	printf( "--------------------------------------------------------------------------------\n" );

	event_output_buffer[0] = '\0';          // start the next event with an empty buffer
	return;
}

static int
parse_event_qualifiers( PAPI_event_info_t * info )
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
	memmove( info->symbol, pmask, ( strlen(pmask) + 1 ) * sizeof(char) );

	//  The description field contains the event description followed by a tag 'masks:'
	//  and then the mask description (if there was a mask with this event).  The following
	//  code isolates the mask description part of this information.

	pmask = strstr( info->long_descr, "masks:" );
	if ( pmask == NULL ) {
		info->long_descr[0] = 0;
	} else {
		pmask += 6;        // bump pointer past 'masks:' identifier in description
		memmove( info->long_descr, pmask, (strlen(pmask) + 1) * sizeof(char) );
	}
	return ( 1 );
}

#if SDE
void
invoke_hook_fptr( char *lib_path )
{
    void *dl_handle;
    typedef void *(* hook_fptr_t)(papi_sde_fptr_struct_t *);
    hook_fptr_t hook_func_ptr;

    /* Clear any old error conditions */
    (void)dlerror();

    dl_handle = dlopen(lib_path, RTLD_LOCAL | RTLD_LAZY);
    if ( NULL == dl_handle ) {
        return;
    }

    hook_func_ptr = (hook_fptr_t)dlsym(dl_handle, "papi_sde_hook_list_events");
    if ( (NULL != hook_func_ptr) && ( NULL == dlerror()) ) {
        papi_sde_fptr_struct_t fptr_struct;

        POPULATE_SDE_FPTR_STRUCT( fptr_struct );
        (void)hook_func_ptr( &fptr_struct );
    }

    dlclose(dl_handle);
    return;
}
#endif

int
main( int argc, char **argv )
{
	int i, k;
	int num_events;
	int num_cmp_events = 0;
	int retval;
	PAPI_event_info_t info;
	const PAPI_hw_info_t *hwinfo = NULL;
	command_flags_t flags;
	int enum_modifier;
	int numcmp, cid;

	/* Initialize before parsing the input arguments */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr, "Error! PAPI_library_init\n");
		return retval;
	}

	/* Parse the command-line arguments */
	parse_args( argc, argv, &flags );

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

	retval = PAPI_set_debug( PAPI_VERB_ECONT );
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error!  PAPI_set_debug\n");
		return retval;
	}

	retval = papi_print_header( "Available native events and hardware information.\n", &hwinfo );
	if ( retval != PAPI_OK ) {
		fprintf(stderr,"Error!  PAPI_get_hardware_info\n");
		return 2;
	}

#if SDE
    /*
       The following code will execute if the user wants to list the SDEs in the
       library (or executable) stored in flags.path. This code will not list the
       SDEs per se, it will only give an opportunity to the library to register
       their SDEs, so they can be listed further down.
    */
    if ( flags.list_sdes ){
        char *cmd;
        FILE *pipe;

        if ( access(flags.path, R_OK) == -1 ){
            fprintf(stderr,"Error!  Unable to read file '%s'.\n",flags.path);
            goto no_sdes;
        }

        int len = 5+strlen(flags.path);
        cmd = (char *)calloc(len, sizeof(char));
        if( NULL == cmd ) goto no_sdes;

        int l = snprintf(cmd, len, "ldd %s",flags.path);
        if(l<len-1){
            free(cmd);
            goto no_sdes;
        }

        /* First open all the dependencies of the file we were given */
        pipe = popen(cmd, "r");
        if( NULL != pipe ){
            while( !feof(pipe) ){
                char *lineptr, *lib_name, *lib_path;
                size_t n=0;
                lineptr = lib_name = lib_path = NULL;

                if( getline(&lineptr, &n, pipe) == -1 ){
                    if(lineptr) free(lineptr);
                    break;
                }

                /* If this line does not give us a path to a library, ignore it. */
                if( (NULL != strstr(lineptr,"not found")) || (NULL == strstr(lineptr," => ")) ) {
                    goto skip_lib;
                }

                int status = sscanf(lineptr, "%ms => %ms (%*x)", &lib_name, &lib_path);
                /* If this line is malformed, ignore it. */
                if(2 != status){
                    /* According to the man page: "it is necessary to call free()
                       only if the scanf() call successfully read a string." */
                    goto skip_lib;
                }

                /* Invoke the hook for the dependency we just discovered */
                invoke_hook_fptr(lib_path);

                if( lib_name ) free(lib_name);
                if( lib_path ) free(lib_path);
skip_lib:
                if(lineptr) free(lineptr);
                lineptr = NULL;
                n=0;
            }
            pclose(pipe);
        }

        /* Finally, invoke the hook for the file the user gave us */
        invoke_hook_fptr(flags.path);

        if( NULL != cmd ) free(cmd);
    }
no_sdes:
#endif //SDE

	/* Do this code if the event name option was specified on the commandline */
	if ( flags.named ) {
		if ( PAPI_event_name_to_code( flags.name, &i ) == PAPI_OK ) {
			if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
				printf( "Event name:     %s\n",	info.symbol);
				printf( "Description:    %s\n", info.long_descr );

				/* handle the PAPI component-style events which have a component:::event type */
				char *ptr;
				if ((ptr=strstr(flags.name, ":::"))) {
					ptr+=3;
					/* handle libpfm4-style events which have a pmu::event type event name */
				} else if ((ptr=strstr(flags.name, "::"))) {
					ptr+=2;
				}
				else {
					ptr=flags.name;
				}

				/* if event qualifiers exist but none specified, process all */
				if ( !strchr( ptr, ':' ) ) {
					if ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) == PAPI_OK ) {
						printf( "\nQualifiers:         Name -- Description\n" );
						do {
							retval = PAPI_get_event_info( i, &info );
							if ( retval == PAPI_OK ) {
								if ( parse_event_qualifiers( &info ) ) {
									printf( "      Info:   %10s -- %s\n", info.symbol, info.long_descr );
								}
							}
						} while ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) == PAPI_OK );
					}
				}
			}
		} else {
			printf("Sorry, an event by the name '%s' could not be found.\n",
					flags.name);
			printf("Is it typed correctly?\n\n");
			exit( 1 );
		}
		return 0;
	}

	// Look at all the events and qualifiers and print the information the user has asked for */

	numcmp = PAPI_num_components(  );

	num_events = 0;

	for ( cid = 0; cid < numcmp; cid++ ) {
		const PAPI_component_info_t *component;
		component=PAPI_get_component_info(cid);

		/* Skip disabled components */
		if (component->disabled && component->disabled != PAPI_EDELAY_INIT) continue;

		printf( "===============================================================================\n" );
		printf( " Native Events in Component: %s\n",component->name);
		printf( "===============================================================================\n" );

		// show this component has not found any events yet
		num_cmp_events = 0;

		/* Always ASK FOR the first event */
		/* Don't just assume it'll be the first numeric value */
		i = 0 | PAPI_NATIVE_MASK;

        retval=PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );

        if (retval==PAPI_OK) {
			do {
				memset( &info, 0, sizeof ( info ) );
				retval = PAPI_get_event_info( i, &info );

                /* This event may not exist */
				if ( retval != PAPI_OK ) continue;

				/* Bail if event name doesn't contain include string */
				if ( flags.include && !strstr( info.symbol, flags.istr ) ) continue;

				/* Bail if event name does contain exclude string */
				if ( flags.xclude && strstr( info.symbol, flags.xstr ) ) continue;

				// if not the first event in this component, put out a divider
				if (num_cmp_events) {
					printf( "--------------------------------------------------------------------------------\n" );
				}

				/* count only events that are actually processed */
				num_events++;
				num_cmp_events++;

				if (flags.check){
					check_event(&info);
				}

				format_event_output( &info, 0);

				/*		modifier = PAPI_NTV_ENUM_GROUPS returns event codes with a
						groups id for each group in which this
						native event lives, in bits 16 - 23 of event code
						terminating with PAPI_ENOEVNT at the end of the list.
						*/

				/* This is an IBM Power issue */
				if ( flags.groups ) {
					k = i;
					if ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_GROUPS, cid ) == PAPI_OK ) {
						printf( "Groups: " );
						do {
							printf( "%4d", ( ( k & PAPI_NTV_GROUP_AND_MASK ) >>
										PAPI_NTV_GROUP_SHIFT ) - 1 );
						} while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_GROUPS, cid ) ==PAPI_OK );
						printf( "\n" );
					}
				}

				// If the user has asked us to check the events then we need to
				// walk the list of qualifiers and try to check the event with each one.
				// Even if the user does not want to display the qualifiers this is necessary
				// to be able to correctly report which events can be used on this system.
				//
				// We also need to walk the list if the user wants to see the qualifiers.

				if (flags.qualifiers || flags.check){
					k = i;
					if ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cid ) == PAPI_OK ) {
						// clear event string using first mask
						char first_event_mask_string[PAPI_HUGE_STR_LEN] = "";

						do {
							retval = PAPI_get_event_info( k, &info );
							if ( retval == PAPI_OK ) {
								// if first event mask string not set yet, set it now
								if (strlen(first_event_mask_string) == 0) {
									strcpy (first_event_mask_string, info.symbol);
								}

								if ( flags.check ) {
									check_event(&info);
								}
								// now test if the event qualifiers should be displayed to the user
								if ( flags.qualifiers ) {
									if ( parse_event_qualifiers( &info ) )
										format_event_output( &info, 2);
								}
							}
						} while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cid ) == PAPI_OK );
						// if we are validating events and the event_available flag is not set yet, try a few more combinations
						if (flags.check  && (event_available == 0)) {
							// try using the event with the first mask defined for the event and the cpu mask
							// this is a kludge but many of the uncore events require an event specific mask (usually
							// the first one defined will do) and they all require the cpu mask
							strcpy (info.symbol, first_event_mask_string);
							strcat (info.symbol, ":cpu=1");
							check_event(&info);
						}
						if (flags.check  && (event_available == 0)) {
							// an even bigger kludge is that there are 4 snpep_unc_pcu events which require the 'ff' and 'cpu' qualifiers to work correctly.
							// if nothing else has worked, this code will try those two qualifiers with the current event name to see if it works
							strcpy (info.symbol, first_event_mask_string);
							char *wptr = strrchr (info.symbol, ':');
							if (wptr != NULL) {
								*wptr = '\0';
								strcat (info.symbol, ":ff=64:cpu=1");
								check_event(&info);
							}
						}
					}
				}
				print_event_output(flags.check);
			} while (PAPI_enum_cmp_event( &i, enum_modifier, cid ) == PAPI_OK );
		}
	}

	if (num_cmp_events != 0) {
		printf( "--------------------------------------------------------------------------------\n" );
	}
	printf( "\nTotal events reported: %d\n", num_events );

	if (num_events==0) {
		printf("\nNo events detected!  Check papi_component_avail to find out why.\n");
		printf("\n");
	}


	return 0;
}
