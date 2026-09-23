/**
 * @file    sde.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is the component for supporting Software Defined Events (SDE).
 *  It provides an interface for libraries, runtimes, and other software
 *  layers to export events to other software layers through PAPI.
 */

#include "sde_internal.h"
#include <string.h>

papi_vector_t _sde_vector;
int _sde_component_lock;

// The following two function pointers will be used by libsde in case PAPI is statically linked (libpapi.a)
void (*papi_sde_check_overflow_status_ptr)(uint32_t cntr_id, long long int value) = &papi_sde_check_overflow_status;
int  (*papi_sde_set_timer_for_overflow_ptr)(void) = &papi_sde_set_timer_for_overflow;

#define DLSYM_CHECK(name)                                              \
    do {                                                               \
        if ( NULL != (err=dlerror()) ) {                               \
            int strErr=snprintf(_sde_vector.cmp_info.disabled_reason,  \
                PAPI_MAX_STR_LEN,                                      \
                "Function '%s' not found in any dynamic lib",          \
                #name);                                                \
            if (strErr > PAPI_MAX_STR_LEN)                             \
                SUBDBG("Unexpected snprintf error.\n");                \
            name##_ptr = NULL;                                         \
            SUBDBG("sde_load_sde_ti(): Unable to load symbol %s: %s\n", #name, err);\
            return ( PAPI_ECMP );                                      \
        }                                                              \
    } while (0)

/*
  If the library is being built statically then there is no need (or ability)
  to access symbols through dlopen/dlsym; applications using the static version
  of PAPI (libpapi.a) must also be linked against libsde for supporting SDEs.
  However, if the dynamic library is used (libpapi.so) then we will look for
  the symbols from libsde.so dynamically.
*/
static int sde_load_sde_ti( void ){
    char *err;

    // In case of static linking the function pointers will be automatically set
    // by the linker and the dlopen()/dlsym() would fail at runtime, so we want to
    // check if the linker has done its magic first.
    if( (NULL != sde_ti_reset_counter_ptr) &&
        (NULL != sde_ti_reset_counter_ptr) &&
        (NULL != sde_ti_read_counter_ptr) &&
        (NULL != sde_ti_write_counter_ptr) &&
        (NULL != sde_ti_name_to_code_ptr) &&
        (NULL != sde_ti_is_simple_counter_ptr) &&
        (NULL != sde_ti_is_counter_set_to_overflow_ptr) &&
        (NULL != sde_ti_set_counter_overflow_ptr) &&
        (NULL != sde_ti_get_event_name_ptr) &&
        (NULL != sde_ti_get_event_description_ptr) &&
        (NULL != sde_ti_get_num_reg_events_ptr) &&
        (NULL != sde_ti_shutdown_ptr)
      ){
        return PAPI_OK;
    }

    (void)dlerror(); // Clear the internal string so we can diagnose errors later on.

    void *handle = dlopen(NULL, RTLD_NOW|RTLD_GLOBAL);
    if( NULL != (err = dlerror()) ){
        SUBDBG("sde_load_sde_ti(): %s\n",err);
        return PAPI_ENOSUPP;
    }

    sde_ti_reset_counter_ptr = (int (*)( uint32_t ))dlsym( handle, "sde_ti_reset_counter" );
    DLSYM_CHECK(sde_ti_reset_counter);

    sde_ti_read_counter_ptr = (int (*)( uint32_t, long long int * ))dlsym( handle, "sde_ti_read_counter" );
    DLSYM_CHECK(sde_ti_read_counter);

    sde_ti_write_counter_ptr = (int (*)( uint32_t, long long ))dlsym( handle, "sde_ti_write_counter" );
    DLSYM_CHECK(sde_ti_write_counter);

    sde_ti_name_to_code_ptr = (int (*)( const char *, uint32_t * ))dlsym( handle, "sde_ti_name_to_code" );
    DLSYM_CHECK(sde_ti_name_to_code);

    sde_ti_is_simple_counter_ptr = (int (*)( uint32_t ))dlsym( handle, "sde_ti_is_simple_counter" );
    DLSYM_CHECK(sde_ti_is_simple_counter);

    sde_ti_is_counter_set_to_overflow_ptr = (int (*)( uint32_t ))dlsym( handle, "sde_ti_is_counter_set_to_overflow" );
    DLSYM_CHECK(sde_ti_is_counter_set_to_overflow);

    sde_ti_set_counter_overflow_ptr = (int (*)( uint32_t, int ))dlsym( handle, "sde_ti_set_counter_overflow" );
    DLSYM_CHECK(sde_ti_set_counter_overflow);

    sde_ti_get_event_name_ptr = (char * (*)( int ))dlsym( handle, "sde_ti_get_event_name" );
    DLSYM_CHECK(sde_ti_get_event_name);

    sde_ti_get_event_description_ptr = (char * (*)( int ))dlsym( handle, "sde_ti_get_event_description" );
    DLSYM_CHECK(sde_ti_get_event_description);

    sde_ti_get_num_reg_events_ptr = (int (*)( void ))dlsym( handle, "sde_ti_get_num_reg_events" );
    DLSYM_CHECK(sde_ti_get_num_reg_events);

    sde_ti_shutdown_ptr = (int (*)( void ))dlsym( handle, "sde_ti_shutdown" );
    DLSYM_CHECK(sde_ti_shutdown);

    return PAPI_OK;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

static int
_sde_init_component( int cidx )
{
    int ret_val = PAPI_OK;
    SUBDBG("_sde_init_component...\n");

    _sde_vector.cmp_info.num_native_events = 0;
    _sde_vector.cmp_info.CmpIdx = cidx;
    _sde_component_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    ret_val = sde_load_sde_ti();
    if( PAPI_OK != ret_val ){
        _sde_vector.cmp_info.disabled = ret_val;
        int expect = snprintf(_sde_vector.cmp_info.disabled_reason,
                              PAPI_MAX_STR_LEN, "libsde API not found. No SDEs exist in this executable.");
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("disabled_reason truncated");
        }
    }

    return ret_val;
}



/** This is called whenever a thread is initialized */
static int
_sde_init_thread( hwd_context_t *ctx )
{
    (void)ctx;
    SUBDBG( "_sde_init_thread %p...\n", ctx );
    return PAPI_OK;
}



/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

static int
_sde_init_control_state( hwd_control_state_t * ctl )
{
    SUBDBG( "sde_init_control_state... %p\n", ctl );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;
    memset( sde_ctl, 0, sizeof ( sde_control_state_t ) );

    return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
static int
_sde_update_control_state( hwd_control_state_t *ctl,
        NativeInfo_t *native,
        int count,
        hwd_context_t *ctx )
{

    (void) ctx;
    int i, index;

    SUBDBG( "_sde_update_control_state %p %p...\n", ctl, ctx );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    for( i = 0; i < count; i++ ) {
        index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
        if( index < 0 ){
            PAPIERROR("_sde_update_control_state(): Event at index %d has a negative native event code = %d.\n",i,index);
            return PAPI_EINVAL;
        }
        SUBDBG("_sde_update_control_state: i=%d index=%u\n", i, index );
        sde_ctl->which_counter[i] = (uint32_t)index;
        native[i].ni_position = i;
    }

    // If an event for which overflowing was set is being removed from the eventset, then the
    // framework will turn overflowing off (by calling PAPI_overflow() with threshold=0),
    // so we don't need to do anything here.

    sde_ctl->num_events=count;

    return PAPI_OK;
}


/** Triggered by PAPI_start() */
static int
_sde_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    int ret_val = PAPI_OK;
    ThreadInfo_t *thread;
    int cidx;
    struct itimerspec its;
    ( void ) ctx;
    ( void ) ctl;

    SUBDBG( "%p %p...\n", ctx, ctl );

    ret_val = _sde_reset(ctx, ctl);

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    its.it_value.tv_sec = 0;
    // We will start the timer at 100us because we adjust its period in _sde_dispatch_timer()
    // if the counter is not growing fast enough, or growing too slowly.
    its.it_value.tv_nsec = 100*1000; // 100us
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );

    if ( (NULL != thread) && (NULL != thread->running_eventset[cidx]) && (thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        if( !(sde_ctl->has_timer) ){
            // No registered counters went through r[1-3]
            int i;
            _papi_hwi_lock(_sde_component_lock);
            for( i = 0; i < sde_ctl->num_events; i++ ) {
                if( sde_ti_is_counter_set_to_overflow_ptr(sde_ctl->which_counter[i]) ){
                    // Registered counters went through r4
                    if( PAPI_OK == do_set_timer_for_overflow(sde_ctl) )
                        break;
                }
            }
            _papi_hwi_unlock(_sde_component_lock);
        }

        // r[1-4]
        if( sde_ctl->has_timer ){
            SUBDBG( "starting SDE internal timer for emulating HARDWARE overflowing\n");
            if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
                PAPIERROR("timer_settime");
                timer_delete(sde_ctl->timerid);
                sde_ctl->has_timer = 0;
                return PAPI_ECMP;
            }
        }
    }

    return ret_val;
}


/** Triggered by PAPI_stop() */
static int
_sde_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

    (void) ctx;
    (void) ctl;
    ThreadInfo_t *thread;
    int cidx;
    struct itimerspec zero_time;

    SUBDBG( "sde_stop %p %p...\n", ctx, ctl );
    /* anything that would need to be done at counter stop time */

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );

    if ( (NULL != thread) && (NULL != thread->running_eventset[cidx]) && (thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        if( sde_ctl->has_timer ){
            SUBDBG( "stopping SDE internal timer\n");
            memset(&zero_time, 0, sizeof(struct itimerspec));
            if (timer_settime(sde_ctl->timerid, 0, &zero_time, NULL) == -1){
                PAPIERROR("timer_settime");
                timer_delete(sde_ctl->timerid);
                sde_ctl->has_timer = 0;
                return PAPI_ECMP;
            }
        }
    }

    return PAPI_OK;
}

/** Triggered by PAPI_read()     */
static int
_sde_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags )
{
    int i;
    int ret_val = PAPI_OK;
    (void) flags;
    (void) ctx;

    SUBDBG( "_sde_read... %p %d\n", ctx, flags );
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    _papi_hwi_lock(_sde_component_lock);
    for( i = 0; i < sde_ctl->num_events; i++ ) {
        uint32_t counter_uniq_id = sde_ctl->which_counter[i];
        ret_val = sde_ti_read_counter_ptr( counter_uniq_id, &(sde_ctl->counter[i]) );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_read(): Error when reading event at index %d.\n",i);
            goto fnct_exit;
        }

    }
    *events = sde_ctl->counter;

fnct_exit:
    _papi_hwi_unlock(_sde_component_lock);
    return ret_val;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
static int
_sde_write( hwd_context_t *ctx, hwd_control_state_t *ctl, long long *values )
{
    int i, ret_val = PAPI_OK;
    (void) ctx;
    (void) ctl;

    SUBDBG( "_sde_write... %p\n", ctx );
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    // Lock before we access global data structures.
    _papi_hwi_lock(_sde_component_lock);
    for( i = 0; i < sde_ctl->num_events; i++ ) {
        uint32_t counter_uniq_id = sde_ctl->which_counter[i];
        ret_val = sde_ti_write_counter_ptr( counter_uniq_id, values[i] );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_write(): Error when writing event at index %d.\n",i);
            goto fnct_exit;
        }
    }

fnct_exit:
    _papi_hwi_unlock(_sde_component_lock);
    return ret_val;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
static int
_sde_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    int i, ret_val=PAPI_OK;
    (void) ctx;

    SUBDBG( "_sde_reset ctx=%p ctrl=%p...\n", ctx, ctl );
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    _papi_hwi_lock(_sde_component_lock);
    for( i = 0; i < sde_ctl->num_events; i++ ) {
        uint32_t counter_uniq_id = sde_ctl->which_counter[i];
        ret_val = sde_ti_reset_counter_ptr( counter_uniq_id );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_reset(): Error when reseting event at index %d.\n",i);
            goto fnct_exit;
        }

    }

fnct_exit:
    _papi_hwi_unlock(_sde_component_lock);
    return ret_val;
}

/** Triggered by PAPI_shutdown() */
static int
_sde_shutdown_component(void)
{
    SUBDBG( "sde_shutdown_component...\n" );
    return sde_ti_shutdown_ptr();
}

/** Called at thread shutdown */
static int
_sde_shutdown_thread( hwd_context_t *ctx )
{

    (void) ctx;

    SUBDBG( "sde_shutdown_thread... %p\n", ctx );

    /* Last chance to clean up thread */

    return PAPI_OK;
}



/** This function sets various options in the component
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
  PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
  */
static int
_sde_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

    (void) ctx;
    (void) code;
    (void) option;

    SUBDBG( "sde_ctl...\n" );

    return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
  In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
  By default return PAPI_EINVAL if none of those are specified
  and PAPI_OK with success
  PAPI_DOM_USER is only user context is counted
  PAPI_DOM_KERNEL is only the Kernel/OS context is counted
  PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
  PAPI_DOM_ALL   is all of the domains
  */
static int
_sde_set_domain( hwd_control_state_t * cntrl, int domain )
{
    (void) cntrl;

    int found = 0;
    SUBDBG( "sde_set_domain...\n" );

    if ( PAPI_DOM_USER & domain ) {
        SUBDBG( " PAPI_DOM_USER\n" );
        found = 1;
    }
    if ( PAPI_DOM_KERNEL & domain ) {
        SUBDBG( " PAPI_DOM_KERNEL\n" );
        found = 1;
    }
    if ( PAPI_DOM_OTHER & domain ) {
        SUBDBG( " PAPI_DOM_OTHER\n" );
        found = 1;
    }
    if ( PAPI_DOM_ALL & domain ) {
        SUBDBG( " PAPI_DOM_ALL\n" );
        found = 1;
    }
    if ( !found )
        return ( PAPI_EINVAL );

    return PAPI_OK;
}


/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/


/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
static int
_sde_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    unsigned int curr_code, next_code, num_reg_events;
    int ret_val = PAPI_OK;

    SUBDBG("_sde_ntv_enum_events begin\n\tEventCode=%u modifier=%d\n", *EventCode, modifier);

    switch ( modifier ) {

        /* return EventCode of first event */
        case PAPI_ENUM_FIRST:
            /* return the first event that we support */
            if( sde_ti_get_num_reg_events_ptr() <= 0 ){
                ret_val = PAPI_ENOEVNT;
                break;
            }
            *EventCode = 0;
            ret_val = PAPI_OK;
            break;

        /* return EventCode of next available event */
        case PAPI_ENUM_EVENTS:
            curr_code = *EventCode & PAPI_NATIVE_AND_MASK;

            // Lock before we read num_reg_events and the hash-tables.
            _papi_hwi_lock(_sde_component_lock);

            num_reg_events = (unsigned int)sde_ti_get_num_reg_events_ptr();
            if( curr_code >= num_reg_events-1 ){
                ret_val = PAPI_ENOEVNT;
                goto unlock;
            }

            /*
             * We have to check the events which follow the current one, because unregistering
             * will create sparcity in the global SDE table, so we can't just return the next
             * index.
             */
            next_code = curr_code;
            do{
                next_code++;
                char *ev_name = sde_ti_get_event_name_ptr((uint32_t)next_code);
                if( NULL != ev_name ){
                    *EventCode = next_code;
                    SUBDBG("Event name = %s (code = %d)\n", ev_name, next_code);
                    ret_val = PAPI_OK;
                    goto unlock;
                }
            }while(next_code < num_reg_events);

            // If we make it here it means that we didn't find the event.
            ret_val = PAPI_EINVAL;

unlock:
            _papi_hwi_unlock(_sde_component_lock);
            break;

        default:
            ret_val = PAPI_EINVAL;
            break;
    }

    return ret_val;
}

/** Takes a native event code and passes back the name
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int
_sde_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int ret_val = PAPI_OK;
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;

    SUBDBG("_sde_ntv_code_to_name %u\n", code);

    _papi_hwi_lock(_sde_component_lock);

    char *ev_name = sde_ti_get_event_name_ptr((uint32_t)code);
    if( NULL == ev_name ){
        ret_val = PAPI_ENOEVNT;
        goto fnct_exit;
    }
    SUBDBG("Event name = %s (code = %d)\n", ev_name, code);
    (void)strncpy( name, ev_name, len );
    name[len-1] = '\0';

fnct_exit:
    _papi_hwi_unlock(_sde_component_lock);
    return ret_val;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int
_sde_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
    int ret_val = PAPI_OK;
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;

    SUBDBG("_sde_ntv_code_to_descr %u\n", code);

    _papi_hwi_lock(_sde_component_lock);

    char *ev_descr = sde_ti_get_event_description_ptr((uint32_t)code);
    if( NULL == ev_descr ){
        ret_val = PAPI_ENOEVNT;
        goto fnct_exit;
    }
    SUBDBG("Event (code = %d) description: %s\n", code, ev_descr);

    (void)strncpy( descr, ev_descr, len );
    descr[len-1] = '\0';

fnct_exit:
    _papi_hwi_unlock(_sde_component_lock);
    return ret_val;
}

/** Takes a native event code and passes back the event info
 * @param EventCode is the native event code
 * @param info is a pointer to a PAPI_event_info_t structure where the event information will be stored.
 */
static int
_sde_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info)
{
    int ret_val = PAPI_OK;
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;

    SUBDBG("_sde_ntv_code_to_info %u\n", code);

	info->event_code = EventCode;
	info->component_index = _sde_vector.cmp_info.CmpIdx;

    _papi_hwi_lock(_sde_component_lock);

    char *ev_name = sde_ti_get_event_name_ptr((uint32_t)code);
    char *ev_descr = sde_ti_get_event_description_ptr((uint32_t)code);
    if( (NULL == ev_descr) || (NULL == ev_name) ){
        ret_val = PAPI_ENOEVNT;
        goto fnct_exit;
    }
    SUBDBG("Event (code = %d) name: %s \n", code, ev_name);
    SUBDBG("Event (code = %d) description: %s\n", code, ev_descr);

    (void)strncpy( info->symbol, ev_name, PAPI_HUGE_STR_LEN-1 );
    info->symbol[PAPI_HUGE_STR_LEN-1] = '\0';

    (void)strncpy( info->long_descr, ev_descr, PAPI_HUGE_STR_LEN-1 );
    info->long_descr[PAPI_HUGE_STR_LEN-1] = '\0';

    (void)strncpy( info->short_descr, ev_descr, PAPI_MIN_STR_LEN-1 );
    info->long_descr[PAPI_MIN_STR_LEN-1] = '\0';

fnct_exit:
    _papi_hwi_unlock(_sde_component_lock);
    return ret_val;
}

/** Takes a native event name and passes back the code
 * @param event_name -- a pointer for the name to be copied to
 * @param event_code -- the native event code
 */
static int
_sde_ntv_name_to_code(const char *event_name, unsigned int *event_code )
{
    int ret_val;

    SUBDBG( "_sde_ntv_name_to_code(%s)\n", event_name );

    ret_val = sde_ti_name_to_code_ptr(event_name, (uint32_t *)event_code);

    return ret_val;
}


static int
_sde_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold ){

    (void)ESI;
    (void)EventIndex;
    (void)threshold;

    SUBDBG("_sde_set_overflow(%d, %d).\n",EventIndex, threshold);

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
    // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
    // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
    int pos = ESI->EventInfoArray[EventIndex].pos[0];
    uint32_t counter_uniq_id = sde_ctl->which_counter[pos];

    // If we still don't know what type the counter is, then we are _not_ in r[1-3] so we can't create a timer here,
    // but we still have to tell the calling tool/app that there was no error, because the timer will be set in the future.
    int ret_val = sde_ti_set_counter_overflow_ptr(counter_uniq_id, threshold);
    if( PAPI_OK >= ret_val ){
        return ret_val;
    }

    // A threshold of zero indicates that overflowing is not needed anymore.
    if( 0 == threshold ){
        // If we had a timer (if the counter was created we wouldn't have one) then delete it.
        if( sde_ctl->has_timer )
            timer_delete(sde_ctl->timerid);
        sde_ctl->has_timer = 0;
    }else{
        // If we are here we are in r[1-3] so we can create the timer
        return do_set_timer_for_overflow(sde_ctl);
    }

    return PAPI_OK;
}

/**
 *  This code assumes that it is called _ONLY_ for registered counters,
 *  and that is why it sets has_timer to REGISTERED_EVENT_MASK
 */
static int do_set_timer_for_overflow( sde_control_state_t *sde_ctl ){
    int signo, sig_offset;
    struct sigevent sigev;
    struct sigaction sa;

    sig_offset = 0;

    // Choose a new real-time signal
    signo = SIGRTMIN+sig_offset;
    if(signo > SIGRTMAX){
        PAPIERROR("do_set_timer_for_overflow(): Unable to create new timer due to large number of existing timers. Overflowing will not be activated for the current event.\n");
        return PAPI_ECMP;
    }

    // setup the signal handler
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = _sde_dispatch_timer;
    sigemptyset(&sa.sa_mask);
    if (sigaction(signo, &sa, NULL) == -1){
        PAPIERROR("do_set_timer_for_overflow(): sigaction() failed.");
        return PAPI_ECMP;
    }

    // create the timer
    sigev.sigev_notify = SIGEV_SIGNAL;
    sigev.sigev_signo = signo;
    sigev.sigev_value.sival_ptr = &(sde_ctl->timerid);
    if (timer_create(CLOCK_REALTIME, &sigev, &(sde_ctl->timerid)) == -1){
        PAPIERROR("do_set_timer_for_overflow(): timer_create() failed.");
        return PAPI_ECMP;
    }
    sde_ctl->has_timer |= REGISTERED_EVENT_MASK;

    return PAPI_OK;
}

static inline int sde_arm_timer(sde_control_state_t *sde_ctl){
    struct itimerspec its;

    // We will start the timer at 100us because we adjust its period in _sde_dispatch_timer()
    // if the counter is not growing fast enough, or growing too slowly.
    its.it_value.tv_sec = 0;
    its.it_value.tv_nsec = 100*1000; // 100us
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;

    SUBDBG( "starting SDE internal timer for emulating HARDWARE overflowing\n");
    if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
        PAPIERROR("timer_settime");
        timer_delete(sde_ctl->timerid);
        sde_ctl->has_timer = 0;

        // If the timer is broken, let the caller know that something internal went wrong.
        return PAPI_ECMP;
    }

    return PAPI_OK;
}

void _sde_dispatch_timer( int n, hwd_siginfo_t *info, void *uc) {

    _papi_hwi_context_t hw_context;
    vptr_t address;
    ThreadInfo_t *thread;
    int i, cidx, retval, isHardware, slow_down, speed_up;
    int found_registered_counters, period_has_changed = 0;
    EventSetInfo_t *ESI;
    struct itimerspec its;
    long long overflow_vector = 0;
    sde_control_state_t *sde_ctl;

    (void) n;

    SUBDBG("SDE timer expired. Dispatching (papi internal) overflow handler\n");

    thread = _papi_hwi_lookup_thread( 0 );
    cidx = _sde_vector.cmp_info.CmpIdx;

    ESI = thread->running_eventset[cidx];
    // This holds only the number of events in the eventset that are set to overflow.
    int event_counter = ESI->overflow.event_counter;
    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    retval = _papi_hwi_read( thread->context[cidx], ESI, ESI->sw_stop );
    if ( retval < PAPI_OK )
        return;

    slow_down = 0;
    speed_up = 0;
    found_registered_counters = 0;
    // Reset the deadline of counters which have exceeded the current deadline
    // and check if we need to slow down the frequency of the timer.
    for ( i = 0; i < event_counter; i++ ) {
        int papi_index = ESI->overflow.EventIndex[i];
        long long deadline, threshold, latest, previous, diff;

        uint32_t counter_uniq_id = sde_ctl->which_counter[papi_index];
        if( !sde_ti_is_simple_counter_ptr( counter_uniq_id ) )
            continue;

        found_registered_counters = 1;

        latest = ESI->sw_stop[papi_index];
        deadline = ESI->overflow.deadline[i];
        threshold = ESI->overflow.threshold[i];

        // Find the increment from the previous measurement.
        previous = sde_ctl->previous_value[papi_index];

        // NOTE: The following code assumes that the counters are "long long". No other
        // NOTE: type will work correctly.
        diff = latest-previous;

        // If it's too small we need to slow down the timer, it it's
        // too large we need to speed up the timer.
        if( 30*diff < threshold ){
            slow_down = 1; // I.e., grow the sampling period
        }else if( 10*diff > threshold ){
            speed_up = 1; // I.e., shrink the sampling period
        }

        // Update the "previous" measurement to be the latest one.
        sde_ctl->previous_value[papi_index] = latest;

        // If this counter has exceeded the deadline, add it in the vector.
        if ( latest >= deadline ) {
            // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
            // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
            // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
            int pos = ESI->EventInfoArray[papi_index].pos[0];
            SUBDBG ( "Event at index %d (and pos %d) has value %lld which exceeds deadline %lld (threshold %lld, accuracy %.2lf)\n",
                     papi_index, pos, latest, deadline, threshold, 100.0*(double)(latest-deadline)/(double)threshold);

            overflow_vector ^= ( long long ) 1 << pos;
            // We adjust the deadline in a way that it remains a multiple of threshold so we don't create an additive error.
            ESI->overflow.deadline[i] = threshold*(latest/threshold) + threshold;
        }
    }

    if( !found_registered_counters && sde_ctl->has_timer ){
        struct itimerspec zero_time;
        memset(&zero_time, 0, sizeof(struct itimerspec));
        if (timer_settime(sde_ctl->timerid, 0, &zero_time, NULL) == -1){
            PAPIERROR("timer_settime");
            timer_delete(sde_ctl->timerid);
            sde_ctl->has_timer = 0;
            return;
        }
        goto no_change_in_period;
    }

    // Since we potentially check multiple counters in the loop above, both conditions could be true (for different counter).
    // In this case, we give speed_up priority.
    if( speed_up )
        slow_down = 0;

    // If neither was set, there is nothing to do here.
    if( !speed_up && !slow_down )
        goto no_change_in_period;

    if( !sde_ctl->has_timer )
        goto no_change_in_period;

    // Get the current value of the timer.
    if( timer_gettime(sde_ctl->timerid, &its) == -1){
        PAPIERROR("timer_gettime() failed. Timer will not be modified.\n");
        goto no_change_in_period;
    }

    period_has_changed = 0;
    // We only reduce the period if it is above 131.6us, so it never drops below 100us.
    if( speed_up && (its.it_interval.tv_nsec > 131607) ){
        double new_val = (double)its.it_interval.tv_nsec;
        new_val /= 1.31607; // sqrt(sqrt(3)) = 1.316074
        its.it_value.tv_nsec = (int)new_val;
        its.it_interval.tv_nsec = its.it_value.tv_nsec;
        period_has_changed = 1;
        SUBDBG ("Timer will be sped up to %ld ns\n", its.it_value.tv_nsec);
    }

    // We only increase the period if it is below 75.9ms, so it never grows above 100ms.
    if( slow_down && (its.it_interval.tv_nsec < 75983800) ){
        double new_val = (double)its.it_interval.tv_nsec;
        new_val *= 1.31607; // sqrt(sqrt(3)) = 1.316074
        its.it_value.tv_nsec = (int)new_val;
        its.it_interval.tv_nsec = its.it_value.tv_nsec;
        period_has_changed = 1;
        SUBDBG ("Timer will be slowed down to %ld ns\n", its.it_value.tv_nsec);
    }

    if( !period_has_changed )
        goto no_change_in_period;

    if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
        PAPIERROR("timer_settime() failed when modifying PAPI internal timer. This might have broken overflow support for this eventset.\n");
        goto no_change_in_period;
    }

no_change_in_period:

    // If none of the events exceeded their deadline, there is nothing else to do.
    if( 0 == overflow_vector ){
        return;
    }

    if ( (NULL== thread) || (NULL == thread->running_eventset[cidx]) || (0 == thread->running_eventset[cidx]->overflow.flags) ){
        PAPIERROR( "_sde_dispatch_timer(): 'Can not access overflow flags'");
        return;
    }

    hw_context.si = info;
    hw_context.ucontext = ( hwd_ucontext_t * ) uc;

    address = GET_OVERFLOW_ADDRESS( hw_context );

    int genOverflowBit = 0;

    _papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context, address, &isHardware, overflow_vector, genOverflowBit, &thread, cidx );

   return;
}

static void invoke_user_handler(uint32_t cntr_uniq_id){
    EventSetInfo_t *ESI;
    int i, cidx;
    ThreadInfo_t *thread;
    sde_control_state_t *sde_ctl;
    _papi_hwi_context_t hw_context;
    ucontext_t uc;
    vptr_t address;
    long long overflow_vector;

    thread = _papi_hwi_lookup_thread( 0 );
    cidx = _sde_vector.cmp_info.CmpIdx;
    ESI = thread->running_eventset[cidx];

    // checking again, just to be sure.
    if( !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        return;
    }

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    // This path comes from papi_sde_inc_counter() which increment _ONLY_ one counter, so we don't
    // need to check if any others have overflown.
    overflow_vector = 0;
    for( i = 0; i < sde_ctl->num_events; i++ ) {
        uint32_t uniq_id = sde_ctl->which_counter[i];

        if( uniq_id == cntr_uniq_id ){
            // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
            // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
            // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
            int pos = ESI->EventInfoArray[i].pos[0];
            if( pos == -1 ){
               PAPIERROR( "The PAPI framework considers this event removed from the eventset, but the component does not\n");
               return;
            }
            overflow_vector = ( long long ) 1 << pos;
        }
    }

    getcontext( &uc );
    hw_context.ucontext = &uc;
    hw_context.si = NULL;
    address = GET_OVERFLOW_ADDRESS( hw_context );

    ESI->overflow.handler( ESI->EventSetIndex, ( void * ) address, overflow_vector, hw_context.ucontext );
    return;
}

void
__attribute__((visibility("default")))
papi_sde_check_overflow_status(uint32_t cntr_uniq_id, long long int latest){
    EventSetInfo_t *ESI;
    int cidx, i, index_in_ESI;
    ThreadInfo_t *thread;
    sde_control_state_t *sde_ctl;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );
    if( NULL == thread )
        return;

    ESI = thread->running_eventset[cidx];
    // Check if there is a running event set and it has some events set to overflow
    if( (NULL == ESI) || !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) )
        return;

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    int event_counter = ESI->overflow.event_counter;

    // Check all the events that are set to overflow
    index_in_ESI = -1;
    for (i = 0; i < event_counter; i++ ) {
        int papi_index = ESI->overflow.EventIndex[i];
        uint32_t uniq_id = sde_ctl->which_counter[papi_index];
        // If the created counter that we are incrementing corresponds to
        // an event that was set to overflow, read the deadline and threshold.
        if( uniq_id == cntr_uniq_id ){
            index_in_ESI = i;
            break;
        }
    }

    if( index_in_ESI >= 0 ){
        long long deadline, threshold;
        deadline = ESI->overflow.deadline[index_in_ESI];
        threshold = ESI->overflow.threshold[index_in_ESI];

        // If the current value has exceeded the deadline then
        // invoke the user handler and update the deadline.
        if( latest > deadline ){
            // We adjust the deadline in a way that it remains a multiple of threshold
            // so we don't create an additive error.
            ESI->overflow.deadline[index_in_ESI] = threshold*(latest/threshold) + threshold;
            invoke_user_handler(cntr_uniq_id);
        }
    }

    return;
}

// The following function should only be called from within
// sde_do_register() in libsde.so, which guarantees we are in cases r[4-6].
int
__attribute__((visibility("default")))
papi_sde_set_timer_for_overflow(void){
    ThreadInfo_t *thread;
    EventSetInfo_t *ESI;
    sde_control_state_t *sde_ctl;

    thread = _papi_hwi_lookup_thread( 0 );
    if( NULL == thread )
        return PAPI_OK;

    // Get the current running eventset and check if it has some events set to overflow.
    int cidx = _sde_vector.cmp_info.CmpIdx;
    ESI = thread->running_eventset[cidx];
    if( (NULL == ESI) || !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) )
        return PAPI_OK;

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    // Below this point we know we have a running eventset, so we are in case r5.
    // Since the event is set to overfow, if there is no timer in the eventset, create one and arm it.
    if( !(sde_ctl->has_timer) ){
        int ret = do_set_timer_for_overflow(sde_ctl);
        if( PAPI_OK != ret ){
            return ret;
        }
        ret = sde_arm_timer(sde_ctl);
        return ret;
    }

    return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _sde_vector = {
    .cmp_info = {
        /* default component information */
        /* (unspecified values are initialized to 0) */
        /* we explicitly set them to zero in this sde */
        /* to show what settings are available            */

        .name = "sde",
        .short_name = "sde",
        .description = "Software Defined Events (SDE) component",
        .version = "1.15",
        .support_version = "n/a",
        .kernel_version = "n/a",
        .num_cntrs =               SDE_MAX_SIMULTANEOUS_COUNTERS,
        .num_mpx_cntrs =           SDE_MAX_SIMULTANEOUS_COUNTERS,
        .default_domain =          PAPI_DOM_USER,
        .available_domains =       PAPI_DOM_USER,
        .default_granularity =     PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig =       PAPI_INT_SIGNAL,
        .hardware_intr =           1,

        /* component specific cmp_info initializations */
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        /* once per thread */
        .context = sizeof ( sde_context_t ),
        /* once per eventset */
        .control_state = sizeof ( sde_control_state_t ),
        /* ?? */
        .reg_value = sizeof ( sde_register_t ),
        /* ?? */
        .reg_alloc = sizeof ( sde_reg_alloc_t ),
    },

    /* function pointers */
    /* by default they are set to NULL */

    /* Used for general PAPI interactions */
    .start =                _sde_start,
    .stop =                 _sde_stop,
    .read =                 _sde_read,
    .reset =                _sde_reset,
    .write =                _sde_write,
    .init_component =       _sde_init_component,
    .init_thread =          _sde_init_thread,
    .init_control_state =   _sde_init_control_state,
    .update_control_state = _sde_update_control_state,
    .ctl =                  _sde_ctl,
    .shutdown_thread =      _sde_shutdown_thread,
    .shutdown_component =   _sde_shutdown_component,
    .set_domain =           _sde_set_domain,
    /* .cleanup_eventset =     NULL, */
    /* called in add_native_events() */
    /* .allocate_registers =   NULL, */

    /* Used for overflow/profiling */
    .dispatch_timer =       _sde_dispatch_timer,
    .set_overflow =         _sde_set_overflow,
    /* .get_overflow_address = NULL, */
    /* .stop_profiling =       NULL, */
    /* .set_profile =          NULL, */

    /* ??? */
    /* .user =                 NULL, */

    /* Name Mapping Functions */
    .ntv_enum_events =   _sde_ntv_enum_events,
    .ntv_code_to_name =  _sde_ntv_code_to_name,
    .ntv_code_to_descr = _sde_ntv_code_to_descr,
    .ntv_code_to_info =  _sde_ntv_code_to_info,
    /* if .ntv_name_to_code not available, PAPI emulates  */
    /* it by enumerating all events and looking manually  */
    .ntv_name_to_code  = _sde_ntv_name_to_code,


    /* These are only used by _papi_hwi_get_native_event_info() */
    /* Which currently only uses the info for printing native   */
    /* event info, not for any sort of internal use.            */
    /* .ntv_code_to_bits =  NULL, */

};

