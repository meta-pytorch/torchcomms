#ifndef SDE_H
#define SDE_H

#ifndef SDE_MAX_SIMULTANEOUS_COUNTERS
#define SDE_MAX_SIMULTANEOUS_COUNTERS 40
#endif

#include <inttypes.h>
#include <dlfcn.h>
#include <assert.h>
#include <time.h>
#include <ucontext.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"

#define REGISTERED_EVENT_MASK 0x2;

/* We do not use this structure, but the framework needs its size */
typedef struct sde_register
{
   int junk;
} sde_register_t;

/* We do not use this structure, but the framework needs its size */
typedef struct sde_reg_alloc
{
	sde_register_t junk;
} sde_reg_alloc_t;

/**
 *  There's one of these per event-set to hold data specific to the EventSet, like
 *  counter start values, number of events in a set and counter uniq ids.
 */
typedef struct sde_control_state
{
  int num_events;
  uint32_t which_counter[SDE_MAX_SIMULTANEOUS_COUNTERS];
  long long counter[SDE_MAX_SIMULTANEOUS_COUNTERS];
  long long previous_value[SDE_MAX_SIMULTANEOUS_COUNTERS];
  timer_t timerid;
  int has_timer;
} sde_control_state_t;

typedef struct sde_context {
   long long junk;
} sde_context_t;


// Function prototypes

static int _sde_reset( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_write( hwd_context_t *ctx, hwd_control_state_t *ctl, long long *events );
static int _sde_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags );
static int _sde_stop( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_start( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_update_control_state( hwd_control_state_t *ctl, NativeInfo_t *native, int count, hwd_context_t *ctx );
static int _sde_init_control_state( hwd_control_state_t * ctl );
static int _sde_init_thread( hwd_context_t *ctx );
static int _sde_init_component( int cidx );
static int _sde_shutdown_component(void);
static int _sde_shutdown_thread( hwd_context_t *ctx );
static int _sde_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option );
static int _sde_set_domain( hwd_control_state_t * cntrl, int domain );
static int _sde_ntv_enum_events( unsigned int *EventCode, int modifier );
static int _sde_ntv_code_to_name( unsigned int EventCode, char *name, int len );
static int _sde_ntv_code_to_descr( unsigned int EventCode, char *descr, int len );
static int _sde_ntv_name_to_code(const char *name, unsigned int *event_code );
static void _sde_dispatch_timer( int n, hwd_siginfo_t *info, void *uc);

static void invoke_user_handler( unsigned int cntr_uniq_id );
static int do_set_timer_for_overflow( sde_control_state_t *sde_ctl );
static inline int sde_arm_timer(sde_control_state_t *sde_ctl);

int papi_sde_lock(void);
int papi_sde_unlock(void);
void papi_sde_check_overflow_status(unsigned int cntr_uniq_id, long long int latest);
int papi_sde_set_timer_for_overflow(void);

// Function pointers that will be initialized by the linker if libpapi and libsde are static (.a)
__attribute__((__common__)) int (*sde_ti_reset_counter_ptr)( uint32_t );
__attribute__((__common__)) int (*sde_ti_read_counter_ptr)( uint32_t, long long int * );
__attribute__((__common__)) int (*sde_ti_write_counter_ptr)( uint32_t, long long );
__attribute__((__common__)) int (*sde_ti_name_to_code_ptr)( const char *, uint32_t * );
__attribute__((__common__)) int (*sde_ti_is_simple_counter_ptr)( uint32_t );
__attribute__((__common__)) int (*sde_ti_is_counter_set_to_overflow_ptr)( uint32_t );
__attribute__((__common__)) int (*sde_ti_set_counter_overflow_ptr)( uint32_t, int );
__attribute__((__common__)) char * (*sde_ti_get_event_name_ptr)( int );
__attribute__((__common__)) char * (*sde_ti_get_event_description_ptr)( int );
__attribute__((__common__)) int (*sde_ti_get_num_reg_events_ptr)( void );
__attribute__((__common__)) int (*sde_ti_shutdown_ptr)( void );

#endif
