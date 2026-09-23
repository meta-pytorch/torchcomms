/**
 * @file    sde_lib_ti.h
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 */

#if !defined(PAPI_SDE_LIB_TI_H)
#define PAPI_SDE_LIB_TI_H

int sde_ti_read_counter( uint32_t counter_id, long long int *rslt_ptr);
int sde_ti_write_counter( uint32_t counter_id, long long value );
int sde_ti_reset_counter( uint32_t counter_id );
int sde_ti_name_to_code(const char *event_name, uint32_t *event_code );
int sde_ti_is_simple_counter(uint32_t counter_id);
int sde_ti_is_counter_set_to_overflow(uint32_t counter_id);
int sde_ti_set_counter_overflow(uint32_t counter_id, int threshold);
char *sde_ti_get_event_name(int event_id);
char *sde_ti_get_event_description(int event_id);
int sde_ti_get_num_reg_events( void );
int sde_ti_shutdown( void );

#endif // !defined(PAPI_SDE_LIB_TI_H)
