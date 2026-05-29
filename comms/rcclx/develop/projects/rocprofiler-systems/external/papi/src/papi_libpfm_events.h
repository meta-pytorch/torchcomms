#ifndef _PAPI_LIBPFM_EVENTS_H
#define _PAPI_LIBPFM_EVENTS_H
#include "papi.h" /* For PAPI_event_info_t */
#include "papi_vector.h" /* For papi_vector_t */

/* 
* File:    papi_libpfm_events.h
*/

/* Prototypes for libpfm name library access */

int _papi_libpfm_error( int pfm_error );
int _papi_libpfm_setup_presets( char *name, int type, int cidx );
int _papi_libpfm_ntv_enum_events( unsigned int *EventCode, int modifier );
int _papi_libpfm_ntv_name_to_code( const char *ntv_name,
				       unsigned int *EventCode );
int _papi_libpfm_ntv_code_to_name( unsigned int EventCode, char *name,
				       int len );
int _papi_libpfm_ntv_code_to_descr( unsigned int EventCode, char *name,
					int len );
int _papi_libpfm_ntv_code_to_bits( unsigned int EventCode,
				       hwd_register_t * bits );
int _papi_libpfm_ntv_code_to_bits_perfctr( unsigned int EventCode,
				       hwd_register_t * bits );
int _papi_libpfm_shutdown(void);
int _papi_libpfm_init(papi_vector_t *my_vector, int cidx);


int _pfm_decode_native_event( unsigned int EventCode, unsigned int *event,
			      unsigned int *umask );
unsigned int _pfm_convert_umask( unsigned int event, unsigned int umask );
int prepare_umask( unsigned int foo, unsigned int *values );
int _papi_libpfm_ntv_code_to_info(unsigned int EventCode, 
                                  PAPI_event_info_t *info);



/* Gross perfctr/perf_events compatability hack */
/* need to think up a better way to handle this */

#ifndef __PERFMON_PERF_EVENT_H__
struct perf_event_attr {
  int config;
  int type;
};

#define PERF_TYPE_RAW 4;

#endif /* !__PERFMON_PERF_EVENT_H__ */


extern int _papi_libpfm_setup_counters( struct perf_event_attr *attr, 
				      hwd_register_t *ni_bits );

#endif // _PAPI_LIBPFM_EVENTS_H
