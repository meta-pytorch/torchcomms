/**
 * @file    sde_lib_ti.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is the tools interface of SDE. It contains the functions that the PAPI
 *  SDE component needs to call in order to access the SDEs inside a library.
 */

#include "sde_lib_internal.h"
#include "sde_lib_lock.h"


// These pointers will not be used anywhere in this code. However, if libpapi.a is linked
// into an application that also links against libsde.a (both static libraries) then the
// linker will use these assignments to set the corrsponding function pointers in libsde.a
__attribute__((__common__)) int (*sde_ti_reset_counter_ptr)( uint32_t )                 = &sde_ti_reset_counter;
__attribute__((__common__)) int (*sde_ti_read_counter_ptr)( uint32_t, long long int * ) = &sde_ti_read_counter;
__attribute__((__common__)) int (*sde_ti_write_counter_ptr)( uint32_t, long long )      = &sde_ti_write_counter;
__attribute__((__common__)) int (*sde_ti_name_to_code_ptr)( const char *, uint32_t * )  = &sde_ti_name_to_code;
__attribute__((__common__)) int (*sde_ti_is_simple_counter_ptr)( uint32_t )             = &sde_ti_is_simple_counter;
__attribute__((__common__)) int (*sde_ti_is_counter_set_to_overflow_ptr)( uint32_t )    = &sde_ti_is_counter_set_to_overflow;
__attribute__((__common__)) int (*sde_ti_set_counter_overflow_ptr)( uint32_t, int )     = &sde_ti_set_counter_overflow;
__attribute__((__common__)) char * (*sde_ti_get_event_name_ptr)( int )                  = &sde_ti_get_event_name;
__attribute__((__common__)) char * (*sde_ti_get_event_description_ptr)( int )           = &sde_ti_get_event_description;
__attribute__((__common__)) int (*sde_ti_get_num_reg_events_ptr)( void )                = &sde_ti_get_num_reg_events;
__attribute__((__common__)) int (*sde_ti_shutdown_ptr)( void )                          = &sde_ti_shutdown;


/*
 *
 */
int
sde_ti_read_counter( uint32_t counter_id, long long int *rslt_ptr){
    int ret_val = SDE_OK;
    papisde_control_t *gctl;

    sde_lock();

    gctl = _papisde_global_control;
    if( NULL == gctl ){
        SDE_ERROR("sde_ti_read_counter(): Attempt to read from unintialized SDE structures.\n");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    if( counter_id >= gctl->num_reg_events ){
        SDE_ERROR("sde_ti_read_counter(): SDE with id %d does not correspond to a registered event.\n",counter_id);
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_id);
    if( NULL == counter ){
        SDE_ERROR("sde_ti_read_counter(): SDE with id %d is clobbered.\n",counter_id);
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    SDEDBG("sde_ti_read_counter(): Reading counter: '%s'.\n",counter->name);

    switch( counter->cntr_class ){
        // If the counter represents a counter group then we need to read the values of all the counters in the group.
        case CNTR_CLASS_GROUP:
            ret_val = sdei_read_counter_group( counter, rslt_ptr );
            if( SDE_OK != ret_val ){
                SDE_ERROR("sde_ti_read_counter(): Error occured when reading counter group: '%s'.\n",counter->name);
            }
            break;

        // Our convention is that read attempts on a placeholder will set the counter to "-1" to
        // signify semantically that there was an error, but the function will not return an error
        // to avoid breaking existing programs that do something funny when an error is returned.
        case CNTR_CLASS_PLACEHOLDER:
            SDEDBG("sde_ti_read_counter(): Attempted read on a placeholder: '%s'.\n",counter->name);
            *rslt_ptr = -1;
            break;

        // If we are not dealing with a simple counter but with a recorder, we need to allocate
        // a contiguous buffer, copy all the recorded data in it, and return to the user a pointer
        // to this buffer cast as a long long.
        case CNTR_CLASS_RECORDER:
            {
            long long used_entries;
            size_t typesize;
            void *out_buffer;

            // At least the first chunk should have been allocated at creation.
            if( NULL == counter->u.cntr_recorder.data->ptr_array[0] ){
                SDE_ERROR( "No space has been allocated for recorder %s\n",counter->name);
                ret_val = SDE_EINVAL;
                break;
            }

            used_entries = counter->u.cntr_recorder.data->used_entries;
            typesize = counter->u.cntr_recorder.data->typesize;

            // NOTE: After returning this buffer we loose track of it, so it's the user's responsibility to free it.
            out_buffer = malloc( used_entries*typesize );
            exp_container_to_contiguous(counter->u.cntr_recorder.data, out_buffer);
            *rslt_ptr = (long long)out_buffer;
            break;
            }

        case CNTR_CLASS_CSET:
            {
            cset_list_object_t *list_head;
            sdei_counting_set_to_list( counter, &list_head );
            *rslt_ptr = (long long)list_head;
            break;
            }

        case CNTR_CLASS_REGISTERED: // fall through
        case CNTR_CLASS_CREATED: // fall through
        case CNTR_CLASS_BASIC: // fall through
        case CNTR_CLASS_CB:
            ret_val = sdei_read_and_update_data_value( counter, counter->previous_data, rslt_ptr );
            if( SDE_OK != ret_val ){
                SDE_ERROR("sde_ti_read_counter(): Error occured when reading counter: '%s'.\n",counter->name);
            }
            break;
    }

fn_exit:
    sde_unlock();
    return ret_val;
}

/*
 *
 */
int
sde_ti_write_counter( uint32_t counter_id, long long value ){
    papisde_control_t *gctl;
    int ret_val = SDE_OK;

    gctl = _papisde_global_control;
    if( NULL == gctl ){
        SDE_ERROR("sde_ti_write_counter(): Attempt to write in unintialized SDE structures.\n");
        return SDE_EINVAL;
    }

    if( counter_id >= gctl->num_reg_events ){
        SDE_ERROR("sde_ti_write_counter(): SDE with id %d does not correspond to a registered event.\n",counter_id);
        return SDE_EINVAL;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_id);
    if( (NULL == counter) || !IS_CNTR_BASIC(counter) ){
        SDE_ERROR("sde_ti_write_counter(): SDE with id %d is clobbered, or a type which does not support writing.\n",counter_id);
        return SDE_EINVAL;
    }

    ret_val = sdei_hardware_write( counter, value );
    if( SDE_OK != ret_val ){
        SDE_ERROR("sde_ti_write_counter(): Error occured when writing counter: '%s'.\n",counter->name);
    }

    return ret_val;
}

/*
 *
 */
int
sde_ti_reset_counter( uint32_t counter_id ){
    int ret_val = SDE_OK;
    papisde_control_t *gctl;

    gctl = _papisde_global_control;
    if( NULL == gctl ){
        SDE_ERROR("sde_ti_reset_counter(): Attempt to modify unintialized SDE structures.\n");
        return SDE_EINVAL;
    }

    if( counter_id >= gctl->num_reg_events ){
        SDE_ERROR("sde_ti_reset_counter(): SDE with id %d does not correspond to a registered event.\n",counter_id);
        return SDE_EINVAL;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_id);
    if( (NULL == counter) || (!IS_CNTR_BASIC(counter) && !IS_CNTR_CALLBACK(counter) && !IS_CNTR_CSET(counter)) ){
        SDEDBG("sde_ti_reset_counter(): SDE with id %d is clobbered, or a type which does not support resetting.\n",counter_id);
        // We allow tools to call this function even if the counter type does not support
        // reseting, so we do not return an error if this is the case.
        return SDE_OK;
    }

    ret_val = sdei_read_and_update_data_value( counter, 0, &(counter->previous_data) );
    if( SDE_OK != ret_val ){
        SDE_ERROR("sde_ti_reset_counter(): Error occured when resetting counter: %s.\n",counter->name);
    }

    return ret_val;
}

/*
 *
 */
int
sde_ti_name_to_code(const char *event_name, uint32_t *event_code ){
    int ret_val;
    papisde_library_desc_t *lib_handle;
    char *pos, *tmp_lib_name;
    sde_counter_t *tmp_item = NULL;
    papisde_control_t *gctl;

    SDEDBG( "%s\n", event_name );

    sde_lock();
    gctl = _papisde_global_control;

    // Let's see if the event has the library name as a prefix (as it should). Note that this is
    // the event name as it comes from the framework, so it should contain the library name, although
    // when the library registers an event counter it will not use the library name as part of the event name.
    tmp_lib_name = strdup(event_name);
    pos = strstr(tmp_lib_name, "::");
    if( NULL != pos ){ // Good, it does.
        *pos = '\0';

        if( NULL == gctl ){
            // If no library has initialized SDEs, and the application is already inquiring
            // about an event, let's initialize SDEs pretending to be the library which corresponds to this event.
            gctl = sdei_get_global_struct();
            lib_handle = do_sde_init(tmp_lib_name, gctl);
            if(NULL == lib_handle){
                SDE_ERROR("sde_ti_name_to_code(): Initialized SDE but unable to register new library: %s\n", tmp_lib_name);
                ret_val = SDE_ECMP;
                goto fn_exit;
            }
        }else{
            int is_library_present = 0;
            // If the library side of the component has been initialized, then look for the library.
            lib_handle = gctl->lib_list_head;
            while(NULL != lib_handle){ // Look for the library.
                if( !strcmp(lib_handle->libraryName, tmp_lib_name) ){
                    // We found the library.
                    is_library_present = 1;
                    // Now, look for the event in the library.
                    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, event_name);
                    break;
                }
                lib_handle = lib_handle->next;
            }

            if( !is_library_present ){
                // If the library side of the component was initialized, but the specific library hasn't called
                // papi_sde_init() then we call it here to allocate the data structures.
                lib_handle = do_sde_init(tmp_lib_name, gctl);
                if(NULL == lib_handle){
                    SDE_ERROR("sde_ti_name_to_code(): Unable to register new library: %s\n", tmp_lib_name);
                    ret_val = SDE_ECMP;
                    goto fn_exit;
                }
             }
        }
        free(tmp_lib_name); // We don't need the library name any more.

        if( NULL != tmp_item ){
            SDEDBG("Found matching counter with global uniq id: %d in library: %s\n", tmp_item->glb_uniq_id, lib_handle->libraryName );
            *event_code = tmp_item->glb_uniq_id;
            ret_val = SDE_OK;
            goto fn_exit;
        } else {
            cntr_class_specific_t cntr_union = {0};
            SDEDBG("Did not find event %s in library %s. Registering a placeholder.\n", event_name, lib_handle->libraryName );

            // Use the current number of registered events as the index of the new one, and increment it.
            uint32_t counter_uniq_id = gctl->num_reg_events++;
            gctl->num_live_events++;

            // At this point in the code "lib_handle" contains a pointer to the data structure for this library whether
            // the actual library has been initialized or not.
            tmp_item = allocate_and_insert(gctl, lib_handle, event_name, counter_uniq_id, PAPI_SDE_RO, PAPI_SDE_long_long, CNTR_CLASS_PLACEHOLDER, cntr_union );
            if(NULL == tmp_item) {
                SDEDBG("Event %s does not exist in library %s and placeholder could not be inserted.\n", event_name, lib_handle->libraryName);
                ret_val = SDE_ECMP;
                goto fn_exit;
            }
            *event_code = tmp_item->glb_uniq_id;
            ret_val = SDE_OK;
            goto fn_exit;
        }
    }else{
        free(tmp_lib_name);
    }

    // If no library has initialized the component and we don't know a library name, then we have to return.
    if( NULL == gctl ){
        ret_val = SDE_ENOEVNT;
        goto fn_exit;
    }

    // If the event name does not have the library name as a prefix, then we need to look in all the libraries for the event. However, in this case
    // we can _not_ register a placeholder because we don't know which library the event belongs to.
    lib_handle = gctl->lib_list_head;
    while(NULL != lib_handle){

        tmp_item = ht_lookup_by_name(lib_handle->lib_counters, event_name);
        if( NULL != tmp_item ){
            *event_code = tmp_item->glb_uniq_id;
            SDEDBG("Found matching counter with global uniq id: %d in library: %s\n", tmp_item->glb_uniq_id, lib_handle->libraryName );
            ret_val = SDE_OK;
            goto fn_exit;
        } else {
            SDEDBG("Failed to find event %s in library %s. Looking in other libraries.\n", event_name, lib_handle->libraryName );
        }

        lib_handle = lib_handle->next;
    }

    ret_val = SDE_ENOEVNT;
fn_exit:
    sde_unlock();
    return ret_val;
}

/*
 *
 */
int
sde_ti_is_simple_counter(uint32_t counter_id){

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl )
        return 0;

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_id);

    if( (NULL == counter) || !IS_CNTR_REGISTERED(counter) )
        return 0;

    return 1;
}

/*
 *
 */
int
sde_ti_is_counter_set_to_overflow(uint32_t counter_id){

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl )
        return 0;

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_id);
    if( (NULL == counter) || !counter->overflow || IS_CNTR_CREATED(counter) )
        return 0;

    return 1;
}

/*
 *
 */
int
sde_ti_set_counter_overflow(uint32_t counter_id, int threshold){

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl )
        return SDE_OK;

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_id);
    // If the counter is created then we will check for overflow every time its value gets updated, we don't need to poll.
    // That is in cases c[1-3]
    if( IS_CNTR_CREATED(counter) )
        return SDE_OK;

    // We do not want to overflow on recorders or counting-sets, because we don't even know what this means.
    if( ( IS_CNTR_RECORDER(counter) || IS_CNTR_CSET(counter) ) && (threshold > 0) ){
        return SDE_EINVAL;
    }

    // If we still don't know what type the counter is, then we are _not_ in r[1-3] so we can't create a timer here.
    if( IS_CNTR_PLACEHOLDER(counter) && (threshold > 0) ){
        SDEDBG("Event is a placeholder (it has not been registered by a library yet), so we cannot start overflow, but we can remember it.\n");
        counter->overflow = 1;
        return SDE_OK;
    }

    if( 0 == threshold ){
        counter->overflow = 0;
    }

    // Return a number higher than SDE_OK (which is zero) to indicate to the caller that the timer needs to be set,
    // because SDE_OK only means that there was no error, but the timer should not be set either because we are dealing
    // with a placeholder, or created counter.
    return 0xFF;
}


/*
 *
 */
char *
sde_ti_get_event_name(int event_id){

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl )
        return NULL;

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, event_id);
    if( NULL == counter )
        return NULL;

    return counter->name;
}

/*
 *
 */
char *
sde_ti_get_event_description(int event_id){

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl )
        return NULL;

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, event_id);
    if( NULL == counter )
        return NULL;

    return counter->description;
}

/*
 *
 */
int
sde_ti_get_num_reg_events( void ){

    papisde_control_t *gctl = _papisde_global_control;
    if( NULL == gctl )
        return 0;

    return gctl->num_reg_events;
}

/*
 *
 */
int
sde_ti_shutdown( void ){
    return SDE_OK;
}
