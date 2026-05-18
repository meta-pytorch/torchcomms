#ifndef EXTRAS_H
#define EXTRAS_H

int _papi_hwi_stop_timer( int timer, int signal );
int _papi_hwi_start_timer( int timer, int signal, int ms );
int _papi_hwi_stop_signal( int signal );
int _papi_hwi_start_signal( int signal, int need_context, int cidx );
int _papi_hwi_initialize( DynamicArray_t ** );
int _papi_hwi_dispatch_overflow_signal( void *papiContext, vptr_t address,
					int *, long long, int,
					ThreadInfo_t ** master, int cidx );
void _papi_hwi_dispatch_profile( EventSetInfo_t * ESI, vptr_t address,
				 long long over, int profile_index );


#endif /* EXTRAS_H */
