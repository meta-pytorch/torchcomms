/*
 * File:    papi_vector.c
 * Author:  Kevin London
 *          london@cs.utk.edu
 * Mods:    Haihang You
 *	       you@cs.utk.edu
 * Mods:    <Your name here>
 *          <Your email here>
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include <string.h>

#ifdef _AIX

/* needed because the get_virt_usec() code uses a hardware context */
/* which is a pmapi definition on AIX.                             */
#include <pmapi.h>
#endif


void
_vectors_error(  )
{
	SUBDBG( "function is not implemented in the component!\n" );
	exit( PAPI_ECMP );
}

int
vec_int_ok_dummy(  )
{
	return PAPI_OK;
}

int
vec_int_one_dummy(  )
{
	return 1;
}

int
vec_int_dummy(  )
{
	return PAPI_ECMP;
}

void *
vec_void_star_dummy(  )
{
	return NULL;
}

void
vec_void_dummy(  )
{
	return;
}

long long
vec_long_long_dummy(  )
{
	return PAPI_ECMP;
}

long long
vec_long_long_context_dummy( hwd_context_t *ignored )
{
        (void) ignored;

	return PAPI_ECMP;
}

char *
vec_char_star_dummy(  )
{
	return NULL;
}

long
vec_long_dummy(  )
{
	return PAPI_ECMP;
}

long long vec_virt_cycles(void)
{
  return ((long long) _papi_os_vector.get_virt_usec() * 
	  _papi_hwi_system_info.hw_info.cpu_max_mhz);
}

long long vec_real_nsec_dummy(void)
{
  return ((long long) _papi_os_vector.get_real_usec() * 1000);

}

long long vec_virt_nsec_dummy(void)
{
  return ((long long) _papi_os_vector.get_virt_usec() * 1000);

}


int
_papi_hwi_innoculate_vector( papi_vector_t * v )
{
	if ( !v )
		return ( PAPI_EINVAL );

	/* component function pointers */
	if ( !v->dispatch_timer )
		v->dispatch_timer =
			( void ( * )( int, hwd_siginfo_t *, void * ) ) vec_void_dummy;
	if ( !v->get_overflow_address )
		v->get_overflow_address =
			( void *( * )( int, char *, int ) ) vec_void_star_dummy;
	if ( !v->start )
		v->start = ( int ( * )( hwd_context_t *, hwd_control_state_t * ) )
			vec_int_dummy;
	if ( !v->stop )
		v->stop = ( int ( * )( hwd_context_t *, hwd_control_state_t * ) )
			vec_int_dummy;
	if ( !v->read )
		v->read = ( int ( * )
					( hwd_context_t *, hwd_control_state_t *, long long **,
					  int ) ) vec_int_dummy;
	if ( !v->reset )
		v->reset = ( int ( * )( hwd_context_t *, hwd_control_state_t * ) )
			vec_int_dummy;
	if ( !v->write )
		v->write =
			( int ( * )( hwd_context_t *, hwd_control_state_t *, long long[] ) )
			vec_int_dummy;
	if ( !v->cleanup_eventset ) 
		v->cleanup_eventset = ( int ( * )( hwd_control_state_t * ) ) vec_int_ok_dummy;
	if ( !v->stop_profiling )
		v->stop_profiling =
			( int ( * )( ThreadInfo_t *, EventSetInfo_t * ) ) vec_int_dummy;
	if ( !v->init_component )
		v->init_component = ( int ( * )( int ) ) vec_int_ok_dummy;
	if ( !v->init_thread )
		v->init_thread = ( int ( * )( hwd_context_t * ) ) vec_int_ok_dummy;
	if ( !v->init_control_state )
		v->init_control_state =
			( int ( * )( hwd_control_state_t * ptr ) ) vec_int_dummy;
	if ( !v->update_control_state )
		v->update_control_state = ( int ( * )
									( hwd_control_state_t *, NativeInfo_t *,
									  int, hwd_context_t * ) ) vec_int_dummy;
	if ( !v->ctl )
		v->ctl = ( int ( * )( hwd_context_t *, int, _papi_int_option_t * ) )
			vec_int_dummy;
	if ( !v->set_overflow )
		v->set_overflow =
			( int ( * )( EventSetInfo_t *, int, int ) ) vec_int_dummy;
	if ( !v->set_profile )
		v->set_profile =
			( int ( * )( EventSetInfo_t *, int, int ) ) vec_int_dummy;

	if ( !v->set_domain )
		v->set_domain =
			( int ( * )( hwd_control_state_t *, int ) ) vec_int_dummy;
	if ( !v->ntv_enum_events )
		v->ntv_enum_events = ( int ( * )( unsigned int *, int ) ) vec_int_dummy;
	if ( !v->ntv_name_to_code )
		v->ntv_name_to_code =
			( int ( * )( const char *, unsigned int * ) ) vec_int_dummy;
	if ( !v->ntv_code_to_name )
		v->ntv_code_to_name =
			( int ( * )( unsigned int, char *, int ) ) vec_int_dummy;
	if ( !v->ntv_code_to_descr )
		v->ntv_code_to_descr =
			( int ( * )( unsigned int, char *, int ) ) vec_int_ok_dummy;
	if ( !v->ntv_code_to_bits )
		v->ntv_code_to_bits =
			( int ( * )( unsigned int, hwd_register_t * ) ) vec_int_dummy;
	if ( !v->ntv_code_to_info )
		v->ntv_code_to_info =
			( int ( * )( unsigned int, PAPI_event_info_t * ) ) vec_int_dummy;

	if ( !v->allocate_registers )
		v->allocate_registers =
			( int ( * )( EventSetInfo_t * ) ) vec_int_ok_dummy;

	if ( !v->shutdown_thread )
		v->shutdown_thread = ( int ( * )( hwd_context_t * ) ) vec_int_dummy;
	if ( !v->shutdown_component )
		v->shutdown_component = ( int ( * )( void ) ) vec_int_ok_dummy;
	if ( !v->user )
		v->user = ( int ( * )( int, void *, void * ) ) vec_int_dummy;
	return PAPI_OK;
}


int
_papi_hwi_innoculate_os_vector( papi_os_vector_t * v )
{
	if ( !v )
		return ( PAPI_EINVAL );

	if ( !v->get_real_cycles )
		v->get_real_cycles = vec_long_long_dummy;
	if ( !v->get_real_usec )
		v->get_real_usec = vec_long_long_dummy;
	if ( !v->get_real_nsec )
		v->get_real_nsec = vec_real_nsec_dummy;
	if ( !v->get_virt_cycles )
		v->get_virt_cycles = vec_virt_cycles;
	if ( !v->get_virt_usec )
		v->get_virt_usec = vec_long_long_dummy;
	if ( !v->get_virt_nsec )
		v->get_virt_nsec = vec_virt_nsec_dummy;

	if ( !v->update_shlib_info )
		v->update_shlib_info = ( int ( * )( papi_mdi_t * ) ) vec_int_dummy;
	if ( !v->get_system_info )
		v->get_system_info = ( int ( * )(  ) ) vec_int_dummy;

	if ( !v->get_memory_info )
		v->get_memory_info =
			( int ( * )( PAPI_hw_info_t *, int ) ) vec_int_dummy;

	if ( !v->get_dmem_info )
		v->get_dmem_info = ( int ( * )( PAPI_dmem_info_t * ) ) vec_int_dummy;

	return PAPI_OK;
}

/* not used?  debug only? */
#if 0
static void *
vector_find_dummy( void *func, char **buf )
{
	void *ptr = NULL;

	if ( vec_int_ok_dummy == ( int ( * )(  ) ) func ) {
		ptr = ( void * ) vec_int_ok_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_int_ok_dummy" );
	} else if ( vec_int_one_dummy == ( int ( * )(  ) ) func ) {
		ptr = ( void * ) vec_int_one_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_int_one_dummy" );
	} else if ( vec_int_dummy == ( int ( * )(  ) ) func ) {
		ptr = ( void * ) vec_int_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_int_dummy" );
	} else if ( vec_void_dummy == ( void ( * )(  ) ) func ) {
		ptr = ( void * ) vec_void_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_void_dummy" );
	} else if ( vec_void_star_dummy == ( void *( * )(  ) ) func ) {
		ptr = ( void * ) vec_void_star_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_void_star_dummy" );
	} else if ( vec_long_long_dummy == ( long long ( * )(  ) ) func ) {
		ptr = ( void * ) vec_long_long_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_long_long_dummy" );
	} else if ( vec_char_star_dummy == ( char *( * )(  ) ) func ) {
		ptr = ( void * ) vec_char_star_dummy;
		*buf = papi_strdup( "vec_char_star_dummy" );
	} else if ( vec_long_dummy == ( long ( * )(  ) ) func ) {
		ptr = ( void * ) vec_long_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_long_dummy" );
	} else {
		ptr = NULL;
	}
	return ( ptr );
}


static void
vector_print_routine( void *func, char *fname, int pfunc )
{
	void *ptr = NULL;
	char *buf = NULL;

	ptr = vector_find_dummy( func, &buf );
	if ( ptr ) {
		printf( "DUMMY: %s is mapped to %s.\n",
				fname, buf );
		papi_free( buf );
	} else if ( ( !ptr && pfunc ) )
		printf( "function: %s is mapped to %p.\n",
				fname, func );
}

static void
vector_print_table( papi_vector_t * v, int print_func )
{

	if ( !v )
		return;

	vector_print_routine( ( void * ) v->dispatch_timer,
						  "_papi_hwd_dispatch_timer", print_func );
	vector_print_routine( ( void * ) v->get_overflow_address,
						  "_papi_hwd_get_overflow_address", print_func );
	vector_print_routine( ( void * ) v->start, "_papi_hwd_start", print_func );
	vector_print_routine( ( void * ) v->stop, "_papi_hwd_stop", print_func );
	vector_print_routine( ( void * ) v->read, "_papi_hwd_read", print_func );
	vector_print_routine( ( void * ) v->reset, "_papi_hwd_reset", print_func );
	vector_print_routine( ( void * ) v->write, "_papi_hwd_write", print_func );
	vector_print_routine( ( void * ) v->cleanup_eventset, 
						  "_papi_hwd_cleanup_eventset", print_func );

	vector_print_routine( ( void * ) v->stop_profiling,
						  "_papi_hwd_stop_profiling", print_func );
	vector_print_routine( ( void * ) v->init_component,
						  "_papi_hwd_init_component", print_func );
	vector_print_routine( ( void * ) v->init_thread, "_papi_hwd_init_thread", print_func );
	vector_print_routine( ( void * ) v->init_control_state,
						  "_papi_hwd_init_control_state", print_func );
	vector_print_routine( ( void * ) v->ctl, "_papi_hwd_ctl", print_func );
	vector_print_routine( ( void * ) v->set_overflow, "_papi_hwd_set_overflow",
						  print_func );
	vector_print_routine( ( void * ) v->set_profile, "_papi_hwd_set_profile",
						  print_func );
	vector_print_routine( ( void * ) v->set_domain, "_papi_hwd_set_domain",
						  print_func );
	vector_print_routine( ( void * ) v->ntv_enum_events,
						  "_papi_hwd_ntv_enum_events", print_func );
	vector_print_routine( ( void * ) v->ntv_name_to_code,
						  "_papi_hwd_ntv_name_to_code", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_name,
						  "_papi_hwd_ntv_code_to_name", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_descr,
						  "_papi_hwd_ntv_code_to_descr", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_bits,
						  "_papi_hwd_ntv_code_to_bits", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_info,
						  "_papi_hwd_ntv_code_to_info", print_func );

	vector_print_routine( ( void * ) v->allocate_registers,
						  "_papi_hwd_allocate_registers", print_func );

	vector_print_routine( ( void * ) v->shutdown_thread, "_papi_hwd_shutdown_thread",
						  print_func );
	vector_print_routine( ( void * ) v->shutdown_component,
						  "_papi_hwd_shutdown_component", print_func );
	vector_print_routine( ( void * ) v->user, "_papi_hwd_user", print_func );
}

#endif
