/**
 * @file    sde_lib_misc.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is a collection of internal utility functions that are needed
 *  to support SDEs.
 */

#include "sde_lib_internal.h"

static int aggregate_value_in_group(long long int *data, long long int *rslt, int cntr_type, int group_flags);
static inline int cast_and_store(void *data, long long int previous_value, void *rslt_ptr, int cntr_type);
static inline int free_counter_resources(sde_counter_t *counter);

int _sde_be_verbose = 0;
int _sde_debug = 0;

static papisde_library_desc_t *find_library_by_name(const char *library_name, papisde_control_t *gctl);
static void insert_library_handle(papisde_library_desc_t *lib_handle, papisde_control_t *gctl);


/*************************************************************************/
/* Utility Functions.                                                    */
/*************************************************************************/

/** sdei_get_global_struct() checks if the global structure has been allocated
    and allocates it if has not.
  @return a pointer to the global structure.
  */
papisde_control_t *sdei_get_global_struct(void){
    // Allocate the global control structure, unless it has already been allocated by another library
    // or the application code calling PAPI_name_to_code() for an SDE.
    if ( !_papisde_global_control ) {
        SDEDBG("sdei_get_global_struct(): global SDE control struct is being allocated.\n");
        _papisde_global_control = (papisde_control_t *)calloc( 1, sizeof( papisde_control_t ) );
    }
    return _papisde_global_control;
}




/** This helper function checks to see if a given library has already been initialized and exists
    in the global structure of the component.
  @param[in] a pointer to the global structure.
  @param[in] a string containing the name of the library.
  @return a pointer to the library handle.
  */
papisde_library_desc_t *find_library_by_name(const char *library_name, papisde_control_t *gctl){

    if( (NULL == gctl) || (NULL == library_name) )
        return NULL;

    papisde_library_desc_t *tmp_lib = gctl->lib_list_head;
    // Check to see if this library has already been initialized.
    while(NULL != tmp_lib){
        char *tmp_name = tmp_lib->libraryName;
        SDEDBG("Checking library: '%s' against registered library: '%s'\n", library_name, tmp_lib->libraryName);
        // If we find the same library already registered, we do not create a new entry.
        if( (NULL != tmp_name) && !strcmp(tmp_name, library_name) )
            return tmp_lib;

        tmp_lib = tmp_lib->next;
    }

    return NULL;
}

/** This helper function simply adds a library handle to the beginning of the list of libraries
    in the global structure. It's only reason of existence is to hide the structure of the
    linked list in case we want to change it in the future.
  @param[in] a pointer to the library handle.
  @param[in] a pointer to the global structure.
  */
void insert_library_handle(papisde_library_desc_t *lib_handle, papisde_control_t *gctl){
    SDEDBG("insert_library_handle(): inserting new handle for library: '%s'\n",lib_handle->libraryName);
    lib_handle->next = gctl->lib_list_head;
    gctl->lib_list_head = lib_handle;

    return;
}


// Initialize library handle, or return the existing one if already
// initialized. This function is _not_ thread safe, so it needs to be called
// from within regions protected by sde_lock()/sde_unlock().
papi_handle_t do_sde_init(const char *name_of_library, papisde_control_t *gctl){

    papisde_library_desc_t *tmp_lib;

    SDEDBG("Registering library: '%s'\n",name_of_library);

    // If the library is already initialized, return the handle to it
    tmp_lib = find_library_by_name(name_of_library, gctl);
    if( NULL != tmp_lib ){
        return tmp_lib;
    }

    // If the library is not already initialized, then initialize it.
    tmp_lib = ( papisde_library_desc_t* ) calloc( 1, sizeof( papisde_library_desc_t ) );
    tmp_lib->libraryName = strdup(name_of_library);

    insert_library_handle(tmp_lib, gctl);

    return tmp_lib;
}

sde_counter_t *allocate_and_insert( papisde_control_t *gctl, papisde_library_desc_t* lib_handle, const char* name, uint32_t uniq_id, int cntr_mode, int cntr_type, enum CNTR_CLASS cntr_class, cntr_class_specific_t cntr_union ){

    // make sure to calloc() the structure, so all the fields which we do not explicitly set remain zero.
    sde_counter_t *item = (sde_counter_t *)calloc(1, sizeof(sde_counter_t));
    if( NULL == item )
        return NULL;

    item->u = cntr_union;
    item->cntr_class = cntr_class;
    item->cntr_type = cntr_type;
    item->cntr_mode = cntr_mode;
    item->glb_uniq_id = uniq_id;
    item->name = strdup( name );
    item->description = strdup( name );
    item->which_lib = lib_handle;

    (void)ht_insert(lib_handle->lib_counters, ht_hash_name(name), item);
    (void)ht_insert(gctl->all_reg_counters, ht_hash_id(uniq_id), item);

    return item;
}

void
sdei_counting_set_to_list( void *cset_handle, cset_list_object_t **list_head )
{
    sde_counter_t *tmp_cset;

    if( NULL == list_head )
        return;

    tmp_cset = (sde_counter_t *)cset_handle;
    if( (NULL == tmp_cset) || !IS_CNTR_CSET(tmp_cset) || (NULL == tmp_cset->u.cntr_cset.data) ){
        SDE_ERROR("sdei_counting_set_to_list(): 'cset_handle' is clobbered.");
        return;
    }

    *list_head = cset_to_list(tmp_cset->u.cntr_cset.data);

    return;
}

// This function modifies data structures, BUT its callers are responsible for aquiring a lock, so it
// is always called in an atomic fashion and thus should not acquire a lock. Actually, locking inside
// this function will cause a deadlock.
int sdei_setup_counter_internals( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, enum CNTR_CLASS cntr_class, cntr_class_specific_t cntr_union )
{
    papisde_library_desc_t *lib_handle;
    sde_counter_t *tmp_item;
    uint32_t counter_uniq_id;
    char *full_event_name;
    int ret_val = SDE_OK;
    int needs_overflow = 0;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        SDE_ERROR("sdei_setup_counter_internals(): 'handle' is clobbered. Unable to register counter.");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    SDEDBG("%s: Counter: '%s' will be added in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);

    if( !is_instant(cntr_mode) && !is_delta(cntr_mode) ){
        SDE_ERROR("Unknown mode %d. SDE counter mode must be either Instant or Delta.",cntr_mode);
        free(full_event_name);
        return SDE_ECMP;
    }

    // Look if the event is already registered.
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);

    if( NULL != tmp_item ){
        if( !IS_CNTR_PLACEHOLDER(tmp_item) ){
            // If it is registered and it is _not_ a placeholder then ignore it silently.
            SDEDBG("%s: Counter: '%s' was already in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);
            free(full_event_name);
            return SDE_OK;
        }
        // If we are here, then it IS a placeholder, so check if we need to start overflowing.
        if( tmp_item->overflow && ( (CNTR_CLASS_REGISTERED == cntr_class) || (CNTR_CLASS_CB == cntr_class) ) ){
            needs_overflow = 1;
        }

        // Since the counter is a placeholder update the mode, the type, and the union that contains the 'data'.
        SDEDBG("%s: Updating placeholder for counter: '%s' in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);

        tmp_item->u = cntr_union;
        tmp_item->cntr_class = cntr_class;
        tmp_item->cntr_mode = cntr_mode;
        tmp_item->cntr_type = cntr_type;
        free(full_event_name);

        return SDE_OK;
    }

    // If neither the event, nor a placeholder exists, then use the current
    // number of registered events as the index of the new one, and increment it.
    papisde_control_t *gctl = sdei_get_global_struct();
    counter_uniq_id = gctl->num_reg_events++;
    gctl->num_live_events++;

    SDEDBG("%s: Counter %s has unique ID = %d\n", __FILE__, full_event_name, counter_uniq_id);

    tmp_item = allocate_and_insert( gctl, lib_handle, full_event_name, counter_uniq_id, cntr_mode, cntr_type, cntr_class, cntr_union );

    if(NULL == tmp_item) {
        SDEDBG("%s: Counter not inserted in SDE %s\n", __FILE__, lib_handle->libraryName);
        free(full_event_name);
        return SDE_ECMP;
    }

    free(full_event_name);

    // Check if we need to worry about overflow (cases r[4-6])
    if( needs_overflow ){
        ret_val = sdei_set_timer_for_overflow();
    }

    return ret_val;
}

int sdei_inc_ref_count(sde_counter_t *counter){
    papisde_list_entry_t *curr;
    if( NULL == counter )
        return SDE_OK;

    // If the counter is a group, recursivelly increment the ref_count of all its children.
    if(CNTR_CLASS_GROUP == counter->cntr_class){
        curr = counter->u.cntr_group.group_head;
        do{
            sde_counter_t *tmp_cntr = curr->item;
            // recursively increment the ref_count of all the elements in the group.
            int ret_val = sdei_inc_ref_count(tmp_cntr);
            if( SDE_OK != ret_val )
                return ret_val;
            curr = curr->next;
        }while(NULL != curr);
    }

    // Increment the ref_count of the counter itself, INCLUDING the case where the counter is a group.
    (counter->ref_count)++;

    return SDE_OK;
}

int sdei_delete_counter(papisde_library_desc_t* lib_handle, const char* name) {
    sde_counter_t *tmp_item;
    papisde_control_t *gctl;
    uint32_t item_uniq_id;
    int ret_val = SDE_OK;

    gctl = sdei_get_global_struct();

    // Look for the counter entry in the hash-table of the library
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, name);
    if( NULL == tmp_item ){
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    if( CNTR_CLASS_GROUP == tmp_item->cntr_class ){
        papisde_list_entry_t *curr, *prev;
        // If we are dealing with a goup, then we need to recurse down all its children and
        // delete them (this might mean free them, or just decrement their ref_count).
        curr = tmp_item->u.cntr_group.group_head;
        prev = curr;
        while(NULL != curr){
            int counter_is_dead = 0;
            sde_counter_t *tmp_cntr = curr->item;
            if( NULL == tmp_cntr ){
                ret_val = SDE_EMISC;
                goto fn_exit;
            }

            // If this counter is going to be freed, we need to remove it from this group.
            if( 0 == tmp_cntr->ref_count )
                counter_is_dead = 1;

            // recursively delete all the elements of the group.
            int ret_val = sdei_delete_counter(lib_handle, tmp_cntr->name);
            if( SDE_OK != ret_val )
                goto fn_exit;

            if( counter_is_dead ){
                if( curr == tmp_item->u.cntr_group.group_head ){
                    // if we were removing with the head, change the head, we can't free() it.
                    tmp_item->u.cntr_group.group_head = curr->next;
                    prev = curr->next;
                    curr = curr->next;
                }else{
                    // if we are removing an element, first bridge the previous to the next.
                    prev->next = curr->next;
                    free(curr);
                    curr = prev->next;
                }
            }else{
                // if we are not removing anything, just move the pointers.
                prev = curr;
                curr = curr->next;
            }
        }
    }

    item_uniq_id = tmp_item->glb_uniq_id;

    // If the reference count is not zero, then we don't remove it from the hash tables
    if( 0 == tmp_item->ref_count ){
        // Delete the entry from the library hash-table (which hashes by name)
        tmp_item = ht_delete(lib_handle->lib_counters, ht_hash_name(name), item_uniq_id);
        if( NULL == tmp_item ){
            ret_val = SDE_EMISC;
            goto fn_exit;
        }

        // Delete the entry from the global hash-table (which hashes by id) and free the memory
        // occupied by the counter (not the hash-table entry 'papisde_list_entry_t', the 'sde_counter_t')
        tmp_item = ht_delete(gctl->all_reg_counters, ht_hash_id(item_uniq_id), item_uniq_id);
        if( NULL == tmp_item ){
            ret_val = SDE_EMISC;
            goto fn_exit;
        }

        // We free the counter only once, although it is in two hash-tables,
        // because it is the same structure that is pointed to by both hash-tables.
        free_counter_resources(tmp_item);

        // Decrement the number of live events.
        (gctl->num_live_events)--;
    }else{
        (tmp_item->ref_count)--;
    }

fn_exit:
    return ret_val;
}

int free_counter_resources(sde_counter_t *counter){
    int i, ret_val = SDE_OK;

    if( NULL == counter )
        return SDE_OK;

    if( 0 == counter->ref_count ){
        switch(counter->cntr_class){
            case CNTR_CLASS_CREATED:
                SDEDBG(" + Freeing Created Counter Data.\n");
                free(counter->u.cntr_basic.data);
                break;
            case CNTR_CLASS_RECORDER:
                SDEDBG(" + Freeing Recorder Data.\n");
                free(counter->u.cntr_recorder.data->sorted_buffer);
                for(i=0; i<EXP_CONTAINER_ENTRIES; i++){
                    free(counter->u.cntr_recorder.data->ptr_array[i]);
                }
                free(counter->u.cntr_recorder.data);
                break;
            case CNTR_CLASS_CSET:
                SDEDBG(" + Freeing CountingSet Data.\n");
                ret_val = cset_delete(counter->u.cntr_cset.data);
                break;
        }

        SDEDBG(" -> Freeing Counter '%s'.\n",counter->name);
        free(counter->name);
        free(counter->description);
        free(counter);
    }

    return ret_val;
}

/**
  This function assumes that all counters in a group (including recursive subgroups) have the same type.
  */
int
sdei_read_counter_group( sde_counter_t *counter, long long int *rslt_ptr ){
    papisde_list_entry_t *curr;
    long long int final_value = 0;

    if( NULL == counter ){
        SDE_ERROR("sdei_read_counter_group(): Counter parameter is NULL.\n");
        return SDE_EINVAL;
    }

    if( !IS_CNTR_GROUP(counter) ){
        SDE_ERROR("sdei_read_counter_group(): Counter '%s' is not a counter group.\n",counter->name);
        return SDE_EINVAL;
    }
    curr = counter->u.cntr_group.group_head;

    do{
        long long int tmp_value = 0;
        int ret_val;

        sde_counter_t *tmp_cntr = curr->item;
        if( NULL == tmp_cntr ){
            SDE_ERROR("sdei_read_counter_group(): List of counters in counter group '%s' is clobbered.\n",counter->name);
            return SDE_EINVAL;
        }

        int read_succesfully = 1;
        // We can _not_ have a recorder inside a group.
        if( IS_CNTR_RECORDER(tmp_cntr) || IS_CNTR_CSET(tmp_cntr) || IS_CNTR_PLACEHOLDER(tmp_cntr) ){
            SDE_ERROR("sdei_read_counter_group(): Counter group contains counter: %s with class: %d.\n",tmp_cntr->name, tmp_cntr->cntr_class);
        }else{
            // We allow counter groups to contain other counter groups recursively.
            if( IS_CNTR_GROUP(tmp_cntr) ){
                ret_val = sdei_read_counter_group( tmp_cntr, &tmp_value );
                if( ret_val != SDE_OK ){
                    // If something went wrong with one counter group, ignore it silently.
                    read_succesfully = 0;
                }
            }else{ // If we are here it means that we are trying to read a real counter.
                ret_val = sdei_read_and_update_data_value( tmp_cntr, tmp_cntr->previous_data, &tmp_value );
                if( SDE_OK != ret_val ){
                    SDE_ERROR("sdei_read_counter_group(): Error occured when reading counter: %s.\n",tmp_cntr->name);
                    read_succesfully = 0;
                }
            }

            if( read_succesfully )
                aggregate_value_in_group(&tmp_value, &final_value, tmp_cntr->cntr_type, counter->u.cntr_group.group_flags);
        }

        curr = curr->next;
    }while(NULL != curr);

    *rslt_ptr = final_value;
    return SDE_OK;
}


/* both "rslt" and "data" are local variables that this component stored after promoting to 64 bits. */
#define _SDE_AGGREGATE( _TYPE, _RSLT_TYPE ) do{\
                switch(group_flags){\
                    case PAPI_SDE_SUM:\
                        *(_RSLT_TYPE *)rslt = (_RSLT_TYPE)  ((_TYPE)(*(_RSLT_TYPE *)rslt) + (_TYPE)(*((_RSLT_TYPE *)data)) );\
                        break;\
                    case PAPI_SDE_MAX:\
                        if( *(_RSLT_TYPE *)rslt < *((_RSLT_TYPE *)data) )\
                            *(_RSLT_TYPE *)rslt = *((_RSLT_TYPE *)data);\
                        break;\
                    case PAPI_SDE_MIN:\
                        if( *(_RSLT_TYPE *)rslt > *((_RSLT_TYPE *)data) )\
                            *(_RSLT_TYPE *)rslt = *((_RSLT_TYPE *)data);\
                        break;\
                    default:\
                        SDEDBG("Unsupported counter group flag: %d\n",group_flags);\
                        return -1;\
                } \
            }while(0)

static int aggregate_value_in_group(long long int *data, long long int *rslt, int cntr_type, int group_flags){
    switch(cntr_type){
        case PAPI_SDE_long_long:
            _SDE_AGGREGATE(long long int, long long int);
            return SDE_OK;
        case PAPI_SDE_int:
            // We need to cast the result to "long long" so it is expanded to 64bit to take up all the space
            _SDE_AGGREGATE(int, long long int);
            return SDE_OK;
        case PAPI_SDE_double:
            _SDE_AGGREGATE(double, double);
            return SDE_OK;
        case PAPI_SDE_float:
            // We need to cast the result to "double" so it is expanded to 64bit to take up all the space
            _SDE_AGGREGATE(float, double);
            return SDE_OK;
        default:
            SDEDBG("Unsupported counter type: %d\n",cntr_type);
            return -1;
    }
}

int
sdei_read_and_update_data_value( sde_counter_t *counter, long long int previous_value, long long int *rslt_ptr ) {
    int ret_val;
    long long int tmp_int;
    void *tmp_data;

    char *event_name = counter->name;

    if( IS_CNTR_BASIC(counter) ){
        SDEDBG("Reading %s by accessing data pointer.\n", event_name);
        tmp_data = counter->u.cntr_basic.data;
    }else if( IS_CNTR_CALLBACK(counter) ){
        SDEDBG("Reading %s by calling registered function pointer.\n", event_name);
        tmp_int = counter->u.cntr_cb.callback(counter->u.cntr_cb.param);
        tmp_data = &tmp_int;
    }else if( IS_CNTR_CSET(counter) ){
        if( 0 == previous_value ){
            SDEDBG("Resetting CountingSet %s by freeing all the elements it contains.\n", event_name);
            return cset_delete(counter->u.cntr_cset.data);
        }else{
            SDEDBG("sdei_read_and_update_data_value(): Event %s is a CountingSet, so it may only be reset by this function.\n", event_name);
            return -1;
        }
    }else{
        SDEDBG("sdei_read_and_update_data_value(): Event %s has neither a variable nor a function pointer associated with it.\n", event_name);
        return -1;
    }

    if( is_instant(counter->cntr_mode) ){
        /* Instant counter means that we don't subtract the previous value (which we read at PAPI_Start()) */
        previous_value = 0;
    } else if( is_delta(counter->cntr_mode) ){
        /* Do nothing here, this is the default mode */
    } else{
        SDEDBG("Unsupported mode (%d) for event: %s\n",counter->cntr_mode, event_name);
        return -1;
    }

    ret_val = cast_and_store(tmp_data, previous_value, rslt_ptr, counter->cntr_type);
    return ret_val;
}

static inline int
cast_and_store(void *data, long long int previous_value, void *rslt_ptr, int cntr_type){
    void *tmp_ptr;

    switch(cntr_type){
        case PAPI_SDE_long_long:
            *(long long int *)rslt_ptr = *((long long int *)data) - previous_value;
            SDEDBG(" value LL=%lld (%lld-%lld)\n", *(long long int *)rslt_ptr, *((long long int *)data), previous_value);
            return SDE_OK;
        case PAPI_SDE_int:
            // We need to cast the result to "long long" so it is expanded to 64bit to take up all the space
            *(long long int *)rslt_ptr = (long long int) (*((int *)data) - (int)previous_value);
            SDEDBG(" value LD=%lld (%d-%d)\n", *(long long int *)rslt_ptr, *((int *)data), (int)previous_value);
            return SDE_OK;
        case PAPI_SDE_double:
            tmp_ptr = &previous_value;
            *(double *)rslt_ptr = (*((double *)data) - *((double *)tmp_ptr));
            SDEDBG(" value LF=%lf (%lf-%lf)\n", *(double *)rslt_ptr, *((double *)data), *((double *)tmp_ptr));
            return SDE_OK;
        case PAPI_SDE_float:
            // We need to cast the result to "double" so it is expanded to 64bit to take up all the space
            tmp_ptr = &previous_value;
            *(double *)rslt_ptr = (double)(*((float *)data) - (float)(*((double *)tmp_ptr)) );
            SDEDBG(" value F=%lf (%f-%f)\n", *(double *)rslt_ptr, *((float *)data), (float)(*((double *)tmp_ptr)) );
            return SDE_OK;
        default:
            SDEDBG("Unsupported counter type: %d\n",cntr_type);
            return -1;
    }

}

int
sdei_hardware_write( sde_counter_t *counter, long long int new_value ){
    double tmp_double;
    void *tmp_ptr;

    switch(counter->cntr_type){
        case PAPI_SDE_long_long:
            *((long long int *)(counter->u.cntr_basic.data)) = new_value;
            break;
        case PAPI_SDE_int:
            *((int *)(counter->u.cntr_basic.data)) = (int)new_value;
            break;
        case PAPI_SDE_double:
            tmp_ptr = &new_value;
            tmp_double = *((double *)tmp_ptr);
            *((double *)(counter->u.cntr_basic.data)) = tmp_double;
            break;
        case PAPI_SDE_float:
            // The pointer has to be 64bit. We can cast the variable to safely convert between bit-widths later on.
            tmp_ptr = &new_value;
            tmp_double = *((double *)tmp_ptr);
            *((float *)(counter->u.cntr_basic.data)) = (float)tmp_double;
            break;
        default:
            SDEDBG("Unsupported counter type: %d\n",counter->cntr_type);
            return -1;
    }

    return SDE_OK;
}
