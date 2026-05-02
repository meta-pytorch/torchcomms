#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "solaris-common.h"

#include <sys/utsname.h>


#if 0
/* once the bug in dladdr is fixed by SUN, (now dladdr caused deadlock when
   used with pthreads) this function can be used again */
int
_solaris_update_shlib_info( papi_mdi_t *mdi )
{
	char fname[80], name[PAPI_HUGE_STR_LEN];
	prmap_t newp;
	int count, t_index;
	FILE *map_f;
	void *vaddr;
	Dl_info dlip;
	PAPI_address_map_t *tmp = NULL;

	sprintf( fname, "/proc/%d/map", getpid(  ) );
	map_f = fopen( fname, "r" );
	if ( !map_f ) {
		PAPIERROR( "fopen(%s) returned < 0", fname );
		return ( PAPI_OK );
	}

	/* count the entries we need */
	count = 0;
	t_index = 0;
	while ( fread( &newp, sizeof ( prmap_t ), 1, map_f ) > 0 ) {
		vaddr = ( void * ) ( 1 + ( newp.pr_vaddr ) );	// map base address 
		if ( dladdr( vaddr, &dlip ) > 0 ) {
			count++;
			if ( ( newp.pr_mflags & MA_EXEC ) && ( newp.pr_mflags & MA_READ ) ) {
				if ( !( newp.pr_mflags & MA_WRITE ) )
					t_index++;
			}
			strcpy( name, dlip.dli_fname );
			if ( strcmp( _papi_hwi_system_info.exe_info.address_info.name,
						 basename( name ) ) == 0 ) {
				if ( ( newp.pr_mflags & MA_EXEC ) &&
					 ( newp.pr_mflags & MA_READ ) ) {
					if ( !( newp.pr_mflags & MA_WRITE ) ) {
						_papi_hwi_system_info.exe_info.address_info.text_start =
							( vptr_t ) newp.pr_vaddr;
						_papi_hwi_system_info.exe_info.address_info.text_end =
							( vptr_t ) ( newp.pr_vaddr + newp.pr_size );
					} else {
						_papi_hwi_system_info.exe_info.address_info.data_start =
							( vptr_t ) newp.pr_vaddr;
						_papi_hwi_system_info.exe_info.address_info.data_end =
							( vptr_t ) ( newp.pr_vaddr + newp.pr_size );
					}
				}
			}
		}

	}
	rewind( map_f );
	tmp =
		( PAPI_address_map_t * ) papi_calloc( t_index - 1,
											  sizeof ( PAPI_address_map_t ) );

	if ( tmp == NULL ) {
		PAPIERROR( "Error allocating shared library address map" );
		return ( PAPI_ENOMEM );
	}
	t_index = -1;
	while ( fread( &newp, sizeof ( prmap_t ), 1, map_f ) > 0 ) {
		vaddr = ( void * ) ( 1 + ( newp.pr_vaddr ) );	// map base address
		if ( dladdr( vaddr, &dlip ) > 0 ) {	// valid name
			strcpy( name, dlip.dli_fname );
			if ( strcmp( _papi_hwi_system_info.exe_info.address_info.name,
						 basename( name ) ) == 0 )
				continue;
			if ( ( newp.pr_mflags & MA_EXEC ) && ( newp.pr_mflags & MA_READ ) ) {
				if ( !( newp.pr_mflags & MA_WRITE ) ) {
					t_index++;
					tmp[t_index].text_start = ( vptr_t ) newp.pr_vaddr;
					tmp[t_index].text_end =
						( vptr_t ) ( newp.pr_vaddr + newp.pr_size );
					strncpy( tmp[t_index].name, dlip.dli_fname,
							 PAPI_HUGE_STR_LEN - 1 );
					tmp[t_index].name[PAPI_HUGE_STR_LEN - 1] = '\0';
				} else {
					if ( t_index < 0 )
						continue;
					tmp[t_index].data_start = ( vptr_t ) newp.pr_vaddr;
					tmp[t_index].data_end =
						( vptr_t ) ( newp.pr_vaddr + newp.pr_size );
				}
			}
		}
	}

	fclose( map_f );

	if ( _papi_hwi_system_info.shlib_info.map )
		papi_free( _papi_hwi_system_info.shlib_info.map );
	_papi_hwi_system_info.shlib_info.map = tmp;
	_papi_hwi_system_info.shlib_info.count = t_index + 1;

	return PAPI_OK;
}
#endif


int
_papi_hwi_init_os(void) {

  struct utsname uname_buffer;

  uname(&uname_buffer);

  strncpy(_papi_os_info.name,uname_buffer.sysname,PAPI_MAX_STR_LEN);

  strncpy(_papi_os_info.version,uname_buffer.release,PAPI_MAX_STR_LEN);

  _papi_os_info.itimer_sig = PAPI_INT_MPX_SIGNAL;
  _papi_os_info.itimer_num = PAPI_INT_ITIMER;
  _papi_os_info.itimer_ns = PAPI_INT_MPX_DEF_US * 1000;
  _papi_os_info.itimer_res_ns = 1;

  return PAPI_OK;
}

#if 0
int
_ultra_hwd_update_shlib_info( papi_mdi_t *mdi )
{
	/*??? system call takes very long */

	char cmd_line[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], fname[L_tmpnam];
	char line[256];
	char address[16], size[10], flags[64], objname[256];
	PAPI_address_map_t *tmp = NULL;

	FILE *f = NULL;
	int t_index = 0, i;
	struct map_record
	{
		long address;
		int size;
		int flags;
		char objname[256];
		struct map_record *next;
	} *tmpr, *head, *curr;

	tmpnam( fname );
	SUBDBG( "Temporary name %s\n", fname );

	sprintf( cmd_line, "/bin/pmap %d > %s", ( int ) getpid(  ), fname );
	if ( system( cmd_line ) != 0 ) {
		PAPIERROR( "Could not run %s to get shared library address map",
				   cmd_line );
		return ( PAPI_OK );
	}

	f = fopen( fname, "r" );
	if ( f == NULL ) {
		PAPIERROR( "fopen(%s) returned < 0", fname );
		remove( fname );
		return ( PAPI_OK );
	}

	/* ignore the first line */
	fgets( line, 256, f );
	head = curr = NULL;
	while ( fgets( line, 256, f ) != NULL ) {
		/* discard the last line */
		if ( strncmp( line, " total", 6 ) != 0 ) {
			sscanf( line, "%s %s %s %s", address, size, flags, objname );
			if ( objname[0] == '/' ) {
				tmpr =
					( struct map_record * )
					papi_malloc( sizeof ( struct map_record ) );
				if ( tmpr == NULL )
					return ( -1 );
				tmpr->next = NULL;
				if ( curr ) {
					curr->next = tmpr;
					curr = tmpr;
				}
				if ( head == NULL ) {
					curr = head = tmpr;
				}

				SUBDBG( "%s\n", objname );

				if ( ( strstr( flags, "read" ) && strstr( flags, "exec" ) ) ||
					 ( strstr( flags, "r" ) && strstr( flags, "x" ) ) ) {
					if ( !( strstr( flags, "write" ) || strstr( flags, "w" ) ) ) {	/* text segment */
						t_index++;
						tmpr->flags = 1;
					} else {
						tmpr->flags = 0;
					}
					sscanf( address, "%lx", &tmpr->address );
					sscanf( size, "%d", &tmpr->size );
					tmpr->size *= 1024;
					strcpy( tmpr->objname, objname );
				}

			}

		}
	}
	tmp =
		( PAPI_address_map_t * ) papi_calloc( t_index - 1,
											  sizeof ( PAPI_address_map_t ) );

	if ( tmp == NULL ) {
		PAPIERROR( "Error allocating shared library address map" );
		return ( PAPI_ENOMEM );
	}

	t_index = -1;
	tmpr = curr = head;
	i = 0;
	while ( curr != NULL ) {
		if ( strcmp( _papi_hwi_system_info.exe_info.address_info.name,
					 basename( curr->objname ) ) == 0 ) {
			if ( curr->flags ) {
				_papi_hwi_system_info.exe_info.address_info.text_start =
					( vptr_t ) curr->address;
				_papi_hwi_system_info.exe_info.address_info.text_end =
					( vptr_t ) ( curr->address + curr->size );
			} else {
				_papi_hwi_system_info.exe_info.address_info.data_start =
					( vptr_t ) curr->address;
				_papi_hwi_system_info.exe_info.address_info.data_end =
					( vptr_t ) ( curr->address + curr->size );
			}
		} else {
			if ( curr->flags ) {
				t_index++;
				tmp[t_index].text_start = ( vptr_t ) curr->address;
				tmp[t_index].text_end =
					( vptr_t ) ( curr->address + curr->size );
				strncpy( tmp[t_index].name, curr->objname,
						 PAPI_HUGE_STR_LEN - 1 );
				tmp[t_index].name[PAPI_HUGE_STR_LEN - 1] = '\0';
			} else {
				if ( t_index < 0 )
					continue;
				tmp[t_index].data_start = ( vptr_t ) curr->address;
				tmp[t_index].data_end =
					( vptr_t ) ( curr->address + curr->size );
			}
		}
		tmpr = curr->next;
		/* free the temporary allocated memory */
		papi_free( curr );
		curr = tmpr;
	}						 /* end of while */

	remove( fname );
	fclose( f );
	if ( _papi_hwi_system_info.shlib_info.map )
		papi_free( _papi_hwi_system_info.shlib_info.map );
	_papi_hwi_system_info.shlib_info.map = tmp;
	_papi_hwi_system_info.shlib_info.count = t_index + 1;

	return ( PAPI_OK );

}

#endif

/* From niagara2 code */
int
_solaris_update_shlib_info( papi_mdi_t *mdi )
{
	char *file = "/proc/self/map";
	char *resolve_pattern = "/proc/self/path/%s";

	char lastobject[PRMAPSZ];
	char link[PAPI_HUGE_STR_LEN];
	char path[PAPI_HUGE_STR_LEN];

	prmap_t mapping;

	int fd, count = 0, total = 0, position = -1, first = 1;
	vptr_t t_min, t_max, d_min, d_max;

	PAPI_address_map_t *pam, *cur;

#ifdef DEBUG
	SUBDBG( "ENTERING FUNCTION  >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	fd = open( file, O_RDONLY );

	if ( fd == -1 ) {
		return PAPI_ESYS;
	}

	memset( lastobject, 0, PRMAPSZ );

#ifdef DEBUG
	SUBDBG( " -> %s: Preprocessing memory maps from procfs\n", __func__ );
#endif

	/* Search through the list of mappings in order to identify a) how many
	   mappings are available and b) how many unique mappings are available. */
	while ( read( fd, &mapping, sizeof ( prmap_t ) ) > 0 ) {
#ifdef DEBUG
		SUBDBG( " -> %s: Found a new memory map entry\n", __func__ );
#endif
		/* Another entry found, just the total count of entries. */
		total++;

		/* Is the mapping accessible and not anonymous? */
		if ( mapping.pr_mflags & ( MA_READ | MA_WRITE | MA_EXEC ) &&
			 !( mapping.pr_mflags & MA_ANON ) ) {
			/* Test if a new library has been found. If a new library has been
			   found a new entry needs to be counted. */
			if ( strcmp( lastobject, mapping.pr_mapname ) != 0 ) {
				strncpy( lastobject, mapping.pr_mapname, PRMAPSZ );
				count++;

#ifdef DEBUG
				SUBDBG( " -> %s: Memory mapping entry valid for %s\n", __func__,
						mapping.pr_mapname );
#endif
			}
		}
	}
#ifdef DEBUG
	SUBDBG( " -> %s: Preprocessing done, starting to analyze\n", __func__ );
#endif


	/* Start from the beginning, now fill in the found mappings */
	if ( lseek( fd, 0, SEEK_SET ) == -1 ) {
		return PAPI_ESYS;
	}

	memset( lastobject, 0, PRMAPSZ );

	/* Allocate memory */
	pam =
		( PAPI_address_map_t * ) papi_calloc( count,
											  sizeof ( PAPI_address_map_t ) );

	while ( read( fd, &mapping, sizeof ( prmap_t ) ) > 0 ) {

		if ( mapping.pr_mflags & MA_ANON ) {
#ifdef DEBUG
			SUBDBG
				( " -> %s: Anonymous mapping (MA_ANON) found for %s, skipping\n",
				  __func__, mapping.pr_mapname );
#endif
			continue;
		}

		/* Check for a new entry */
		if ( strcmp( mapping.pr_mapname, lastobject ) != 0 ) {
#ifdef DEBUG
			SUBDBG( " -> %s: Analyzing mapping for %s\n", __func__,
					mapping.pr_mapname );
#endif
			cur = &( pam[++position] );
			strncpy( lastobject, mapping.pr_mapname, PRMAPSZ );
			snprintf( link, PAPI_HUGE_STR_LEN, resolve_pattern, lastobject );
			memset( path, 0, PAPI_HUGE_STR_LEN );
			readlink( link, path, PAPI_HUGE_STR_LEN );
			strncpy( cur->name, path, PAPI_HUGE_STR_LEN );
#ifdef DEBUG
			SUBDBG( " -> %s: Resolved name for %s: %s\n", __func__,
					mapping.pr_mapname, cur->name );
#endif
		}

		if ( mapping.pr_mflags & MA_READ ) {
			/* Data (MA_WRITE) or text (MA_READ) segment? */
			if ( mapping.pr_mflags & MA_WRITE ) {
				cur->data_start = ( vptr_t ) mapping.pr_vaddr;
				cur->data_end =
					( vptr_t ) ( mapping.pr_vaddr + mapping.pr_size );

				if ( strcmp
					 ( cur->name,
					   _papi_hwi_system_info.exe_info.fullname ) == 0 ) {
					_papi_hwi_system_info.exe_info.address_info.data_start =
						cur->data_start;
					_papi_hwi_system_info.exe_info.address_info.data_end =
						cur->data_end;
				}

				if ( first )
					d_min = cur->data_start;
				if ( first )
					d_max = cur->data_end;

				if ( cur->data_start < d_min ) {
					d_min = cur->data_start;
				}

				if ( cur->data_end > d_max ) {
					d_max = cur->data_end;
				}
			} else if ( mapping.pr_mflags & MA_EXEC ) {
				cur->text_start = ( vptr_t ) mapping.pr_vaddr;
				cur->text_end =
					( vptr_t ) ( mapping.pr_vaddr + mapping.pr_size );

				if ( strcmp
					 ( cur->name,
					   _papi_hwi_system_info.exe_info.fullname ) == 0 ) {
					_papi_hwi_system_info.exe_info.address_info.text_start =
						cur->text_start;
					_papi_hwi_system_info.exe_info.address_info.text_end =
						cur->text_end;
				}

				if ( first )
					t_min = cur->text_start;
				if ( first )
					t_max = cur->text_end;

				if ( cur->text_start < t_min ) {
					t_min = cur->text_start;
				}

				if ( cur->text_end > t_max ) {
					t_max = cur->text_end;
				}
			}
		}

		first = 0;
	}

	close( fd );

	/* During the walk of shared objects the upper and lower bound of the
	   segments could be discovered. The bounds are stored in the PAPI info
	   structure. The information is important for the profiling functions of
	   PAPI. */

/* This variant would pass the addresses of all text and data segments 
  _papi_hwi_system_info.exe_info.address_info.text_start = t_min;
  _papi_hwi_system_info.exe_info.address_info.text_end = t_max;
  _papi_hwi_system_info.exe_info.address_info.data_start = d_min;
  _papi_hwi_system_info.exe_info.address_info.data_end = d_max;
*/

#ifdef DEBUG
	SUBDBG( " -> %s: Analysis of memory maps done, results:\n", __func__ );
	SUBDBG( " -> %s: text_start=%#x, text_end=%#x, text_size=%lld\n", __func__,
			_papi_hwi_system_info.exe_info.address_info.text_start,
			_papi_hwi_system_info.exe_info.address_info.text_end,
			_papi_hwi_system_info.exe_info.address_info.text_end
			- _papi_hwi_system_info.exe_info.address_info.text_start );
	SUBDBG( " -> %s: data_start=%#x, data_end=%#x, data_size=%lld\n", __func__,
			_papi_hwi_system_info.exe_info.address_info.data_start,
			_papi_hwi_system_info.exe_info.address_info.data_end,
			_papi_hwi_system_info.exe_info.address_info.data_end
			- _papi_hwi_system_info.exe_info.address_info.data_start );
#endif

	/* Store the map read and the total count of shlibs found */
	_papi_hwi_system_info.shlib_info.map = pam;
	_papi_hwi_system_info.shlib_info.count = count;

#ifdef DEBUG
	SUBDBG( "LEAVING FUNCTION  >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	return PAPI_OK;
}

#if 0
int
_niagara2_get_system_info( papi_mdi_t *mdi )
{
	// Used for evaluating return values
	int retval = 0;
	// Check for process settings
	pstatus_t *proc_status;
	psinfo_t *proc_info;
	// Used for string truncating
	char *c_ptr;
	// For retrieving the executable full name
	char exec_name[PAPI_HUGE_STR_LEN];
	// For retrieving processor information
	__sol_processor_information_t cpus;

#ifdef DEBUG
	SUBDBG( "ENTERING FUNCTION >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	/* Get and set pid */
	pid = getpid(  );

	/* Check for microstate accounting */
	proc_status = __sol_get_proc_status( pid );

	if ( proc_status->pr_flags & PR_MSACCT == 0 ||
		 proc_status->pr_flags & PR_MSFORK == 0 ) {
		/* Solaris 10 should have microstate accounting always activated */
		return PAPI_ECMP;
	}

	/* Fill _papi_hwi_system_info.exe_info.fullname */
	proc_info = __sol_get_proc_info( pid );

	// If there are arguments, trim the string to the executable name.
	if ( proc_info->pr_argc > 1 ) {
		c_ptr = strchr( proc_info->pr_psargs, ' ' );
		if ( c_ptr != NULL )
			c_ptr = '\0';
	}

	/* If the path can be qualified, use the full path, otherwise the trimmed
	   name. */
	if ( realpath( proc_info->pr_psargs, exec_name ) != NULL ) {
		strncpy( _papi_hwi_system_info.exe_info.fullname, exec_name,
				 PAPI_HUGE_STR_LEN );
	} else {
		strncpy( _papi_hwi_system_info.exe_info.fullname, proc_info->pr_psargs,
				 PAPI_HUGE_STR_LEN );
	}

	/* Fill _papi_hwi_system_info.exe_info.address_info */
	// Taken from the old component
	strncpy( _papi_hwi_system_info.exe_info.address_info.name,
			 basename( _papi_hwi_system_info.exe_info.fullname ),
			 PAPI_HUGE_STR_LEN );
	__CHECK_ERR_PAPI( _niagara2_update_shlib_info( &_papi_hwi_system_info ) );

	/* Fill _papi_hwi_system_info.hw_info */

	// Taken from the old component
	_papi_hwi_system_info.hw_info.ncpu = sysconf( _SC_NPROCESSORS_ONLN );
	_papi_hwi_system_info.hw_info.nnodes = 1;
	_papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_SUN;
	strcpy( _papi_hwi_system_info.hw_info.vendor_string, "SUN" );
	_papi_hwi_system_info.hw_info.totalcpus = sysconf( _SC_NPROCESSORS_CONF );
	_papi_hwi_system_info.hw_info.model = 1;
	strcpy( _papi_hwi_system_info.hw_info.model_string, cpc_cciname( cpc ) );

	/* The field sparc-version is no longer in prtconf -pv */
	_papi_hwi_system_info.hw_info.revision = 1;

	/* Clock speed */
	_papi_hwi_system_info.hw_info.mhz = ( float ) __sol_get_processor_clock(  );
	_papi_hwi_system_info.hw_info.clock_mhz = __sol_get_processor_clock(  );
	_papi_hwi_system_info.hw_info.cpu_max_mhz = __sol_get_processor_clock(  );
	_papi_hwi_system_info.hw_info.cpu_min_mhz = __sol_get_processor_clock(  );

	/* Fill _niagara2_vector.cmp_info.mem_hierarchy */

	_niagara2_get_memory_info( &_papi_hwi_system_info.hw_info, 0 );

	/* Fill _papi_hwi_system_info.sub_info */
	strcpy( _niagara2_vector.cmp_info.name, "SunNiagara2" );
	strcpy( _niagara2_vector.cmp_info.version, "ALPHA" );
	strcpy( _niagara2_vector.cmp_info.support_version, "libcpc2" );
	strcpy( _niagara2_vector.cmp_info.kernel_version, "libcpc2" );

	/* libcpc2 uses SIGEMT using real hardware signals, no sw emu */

#ifdef DEBUG
	SUBDBG( "LEAVING FUNCTION  >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	return PAPI_OK;
}

#endif

int
_solaris_get_system_info( papi_mdi_t *mdi )
{
	int retval;
	pid_t pid;
	char maxargs[PAPI_MAX_STR_LEN] = "<none>";
	psinfo_t psi;
	int fd;
	int hz, version;
	char cpuname[PAPI_MAX_STR_LEN], pname[PAPI_HUGE_STR_LEN];

	/* Check counter access */

	if ( cpc_version( CPC_VER_CURRENT ) != CPC_VER_CURRENT )
		return PAPI_ECMP;
	SUBDBG( "CPC version %d successfully opened\n", CPC_VER_CURRENT );

	if ( cpc_access(  ) == -1 )
		return PAPI_ECMP;

	/* Global variable cpuver */

	cpuver = cpc_getcpuver(  );
	SUBDBG( "Got %d from cpc_getcpuver()\n", cpuver );
	if ( cpuver == -1 )
		return PAPI_ECMP;

#ifdef DEBUG
	{
		if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
			const char *name;
			int i;

			name = cpc_getcpuref( cpuver );
			if ( name ) {
				SUBDBG( "CPC CPU reference: %s\n", name );
			}
			else {
				SUBDBG( "Could not get a CPC CPU reference\n" );
			}

			for ( i = 0; i < cpc_getnpic( cpuver ); i++ ) {
				SUBDBG( "\n%6s %-40s %8s\n", "Reg", "Symbolic name", "Code" );
				cpc_walk_names( cpuver, i, "%6d %-40s %02x\n",
								print_walk_names );
			}
			SUBDBG( "\n" );
		}
	}
#endif


	/* Initialize other globals */

	if ( ( retval = build_tables(  ) ) != PAPI_OK )
		return retval;

	preset_search_map = preset_table;
	if ( cpuver <= CPC_ULTRA2 ) {
		SUBDBG( "cpuver (==%d) <= CPC_ULTRA2 (==%d)\n", cpuver, CPC_ULTRA2 );
		pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT;
		pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT;
	} else if ( cpuver <= LASTULTRA3 ) {
		SUBDBG( "cpuver (==%d) <= CPC_ULTRA3x (==%d)\n", cpuver, LASTULTRA3 );
		pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT;
		pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT;
		_solaris_vector.cmp_info.hardware_intr = 1;
		_solaris_vector.cmp_info.hardware_intr_sig = SIGEMT;
	} else
		return PAPI_ECMP;

	/* Path and args */

	pid = getpid(  );
	if ( pid == -1 )
		return ( PAPI_ESYS );

	/* Turn on microstate accounting for this process and any LWPs. */

	sprintf( maxargs, "/proc/%d/ctl", ( int ) pid );
	if ( ( fd = open( maxargs, O_WRONLY ) ) == -1 )
		return ( PAPI_ESYS );
	{
		int retval;
		struct
		{
			long cmd;
			long flags;
		} cmd;
		cmd.cmd = PCSET;
		cmd.flags = PR_MSACCT | PR_MSFORK;
		retval = write( fd, &cmd, sizeof ( cmd ) );
		close( fd );
		SUBDBG( "Write PCSET returned %d\n", retval );
		if ( retval != sizeof ( cmd ) )
			return ( PAPI_ESYS );
	}

	/* Get executable info */

	sprintf( maxargs, "/proc/%d/psinfo", ( int ) pid );
	if ( ( fd = open( maxargs, O_RDONLY ) ) == -1 )
		return ( PAPI_ESYS );
	read( fd, &psi, sizeof ( psi ) );
	close( fd );

	/* Cut off any arguments to exe */
	{
		char *tmp;
		tmp = strchr( psi.pr_psargs, ' ' );
		if ( tmp != NULL )
			*tmp = '\0';
	}

	if ( realpath( psi.pr_psargs, pname ) )
		strncpy( _papi_hwi_system_info.exe_info.fullname, pname,
				 PAPI_HUGE_STR_LEN );
	else
		strncpy( _papi_hwi_system_info.exe_info.fullname, psi.pr_psargs,
				 PAPI_HUGE_STR_LEN );

	/* please don't use pr_fname here, because it can only store less that 
	   16 characters */
	strcpy( _papi_hwi_system_info.exe_info.address_info.name,
			basename( _papi_hwi_system_info.exe_info.fullname ) );

	SUBDBG( "Full Executable is %s\n",
			_papi_hwi_system_info.exe_info.fullname );

	/* Executable regions, reading /proc/pid/maps file */
	retval = _ultra_hwd_update_shlib_info( &_papi_hwi_system_info );

	/* Hardware info */

	_papi_hwi_system_info.hw_info.ncpu = sysconf( _SC_NPROCESSORS_ONLN );
	_papi_hwi_system_info.hw_info.nnodes = 1;
	_papi_hwi_system_info.hw_info.totalcpus = sysconf( _SC_NPROCESSORS_CONF );

	retval = scan_prtconf( cpuname, PAPI_MAX_STR_LEN, &hz, &version );
	if ( retval == -1 )
		return PAPI_ECMP;

	strcpy( _papi_hwi_system_info.hw_info.model_string,
			cpc_getcciname( cpuver ) );
	_papi_hwi_system_info.hw_info.model = cpuver;
	strcpy( _papi_hwi_system_info.hw_info.vendor_string, "SUN" );
	_papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_SUN;
	_papi_hwi_system_info.hw_info.revision = version;

	_papi_hwi_system_info.hw_info.mhz = ( ( float ) hz / 1.0e6 );
	SUBDBG( "hw_info.mhz = %f\n", _papi_hwi_system_info.hw_info.mhz );

	_papi_hwi_system_info.hw_info.cpu_max_mhz = _papi_hwi_system_info.hw_info.mhz;
	_papi_hwi_system_info.hw_info.cpu_min_mhz = _papi_hwi_system_info.hw_info.mhz;


	/* Number of PMCs */

	retval = cpc_getnpic( cpuver );
	if ( retval < 0 )
		return PAPI_ECMP;

	_solaris_vector.cmp_info.num_cntrs = retval;
	_solaris_vector.cmp_info.fast_real_timer = 1;
	_solaris_vector.cmp_info.fast_virtual_timer = 1;
	_solaris_vector.cmp_info.default_domain = PAPI_DOM_USER;
	_solaris_vector.cmp_info.available_domains =
		PAPI_DOM_USER | PAPI_DOM_KERNEL;

	/* Setup presets */

	retval = _papi_hwi_setup_all_presets( preset_search_map, NULL );
	if ( retval )
		return ( retval );

	return ( PAPI_OK );
}


long long
_solaris_get_real_usec( void )
{
	return ( ( long long ) gethrtime(  ) / ( long long ) 1000 );
}

long long
_solaris_get_real_cycles( void )
{
	return ( _ultra_hwd_get_real_usec(  ) *
			 ( long long ) _papi_hwi_system_info.hw_info.cpu_max_mhz );
}

long long
_solaris_get_virt_usec( void )
{
	return ( ( long long ) gethrvtime(  ) / ( long long ) 1000 );
}





