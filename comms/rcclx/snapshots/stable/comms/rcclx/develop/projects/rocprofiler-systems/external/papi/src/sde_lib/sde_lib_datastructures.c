/**
 * @file    sde_lib_datastructures.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is a collection of functions that manipulate datastructures
 *  that are used by libsde.
 */

#include "sde_lib_internal.h"

/******************************************************************************/
/* Functions related to the hash-table used for internal hashing of events.   */
/******************************************************************************/
uint32_t ht_hash_id(uint32_t uniq_id){
    return uniq_id%PAPISDE_HT_SIZE;
}

// djb2 hash
uint32_t ht_hash_name(const char *str)
{
    uint32_t hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash % PAPISDE_HT_SIZE;
}

void ht_insert(papisde_list_entry_t *hash_table, int ht_key, sde_counter_t *sde_counter)
{
    papisde_list_entry_t *list_head, *new_entry;

    list_head = &hash_table[ht_key];
    // If we have no counter is associated with this key we will put the new
    // counter on the head of the list which has already been allocated.
    if( NULL == list_head->item ){
        list_head->item = sde_counter;
        list_head->next = NULL; // Just for aesthetic reasons.
        return;
    }

    // If we made it here it means that the head was occupied, so we
    // will allocate a new element and put it just after the head.
    new_entry = (papisde_list_entry_t *)calloc(1, sizeof(papisde_list_entry_t));
    new_entry->item = sde_counter;
    new_entry->next = list_head->next;
    list_head->next = new_entry;

    return;
}

// This function serializes the items contained in the hash-table. A pointer
// to the resulting serialized array is put into the parameter "rslt_array".
// The return value indicates the size of the array.
// The array items are copies of the hash-table item into newly allocated
// memory. They do not reference the original items in the hash table. However,
// this is a shallow copy. If the items contain pointers, then the pointed
// elements are _NOT_ copied.
// The caller is responsible for freeing the resulting array memory.
int ht_to_array(papisde_list_entry_t *hash_table, sde_counter_t **rslt_array)
{
    int i, item_cnt = 0, index=0;
    papisde_list_entry_t *list_head, *curr;

    // First pass counts how many items have been inserted in the hash table.

    // Traverse all the elements of the hash-table.
    for(i=0; i<PAPISDE_HT_SIZE; i++){
        // For each element traverse the linked-list starting there (if any).
        list_head = &(hash_table[i]);

        if(NULL != list_head->item){
            item_cnt++;
        }
        for(curr = list_head->next; NULL != curr; curr=curr->next){
            if(NULL == curr->item){ // This can only legally happen for the head of the list.
                SDE_ERROR("ht_to_array(): the hash table is clobbered.");
            }else{
                item_cnt++;
            }
        }
    }

    // Allocate a contiguous array to store the items.
    sde_counter_t *array = (sde_counter_t *)malloc( item_cnt * sizeof(sde_counter_t));

    // Traverse the hash-table again and copy all the items to the array we just allocated.
    for(i=0; i<PAPISDE_HT_SIZE; i++){
        list_head = &(hash_table[i]);

        if(NULL != list_head->item){
            memcpy( &array[index], list_head->item, sizeof(sde_counter_t) );
            index++;
        }
        for(curr = list_head->next; NULL != curr; curr=curr->next){
            if(NULL == curr->item){ // This can only legally happen for the head of the list.
                SDE_ERROR("ht_to_array(): the hash table is clobbered.");
            }else{
                memcpy( &array[index], curr->item, sizeof(sde_counter_t) );
                index++;
            }
        }
    }
    *rslt_array = array;

    return item_cnt;
}

sde_counter_t *ht_delete(papisde_list_entry_t *hash_table, int ht_key, uint32_t uniq_id)
{
    papisde_list_entry_t *list_head, *curr, *prev;
    sde_counter_t *item;

    list_head = &hash_table[ht_key];
    if( NULL == list_head->item ){
        SDE_ERROR("ht_delete(): the entry does not exist.");
        return NULL;
    }

    // If the head contains the element to be deleted, free the space of the counter and pull the list up.
    if( list_head->item->glb_uniq_id == uniq_id ){
        item = list_head->item;
        if( NULL != list_head->next){
            *list_head = *(list_head->next);
        }else{
            memset(list_head, 0, sizeof(papisde_list_entry_t));
        }
        return item;
    }

    prev = list_head;
    // Traverse the linked list to find the element.
    for(curr=list_head->next; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This is only permitted for the head of the list.
            SDE_ERROR("ht_delete(): the hash table is clobbered.");
            return NULL;
        }
        if(curr->item->glb_uniq_id == uniq_id){
            prev->next = curr->next;
            item = curr->item;
            free(curr); // free the hash table entry
            return item;
        }
        prev = curr;
    }

    SDE_ERROR("ht_delete(): the item is not in the list.");
    return NULL;
}

sde_counter_t *ht_lookup_by_name(papisde_list_entry_t *hash_table, const char *name)
{
    papisde_list_entry_t *list_head, *curr;

    list_head = &hash_table[ht_hash_name(name)];
    if( NULL == list_head->item ){
        return NULL;
    }

    for(curr=list_head; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This can only legally happen for the head of the list.
            SDE_ERROR("ht_lookup_by_name() the hash table is clobbered.");
            return NULL;
        }
        if( !strcmp(curr->item->name, name) ){
            return curr->item;
        }
    }

    return NULL;
}

sde_counter_t *ht_lookup_by_id(papisde_list_entry_t *hash_table, uint32_t uniq_id)
{
    papisde_list_entry_t *list_head, *curr;

    list_head = &hash_table[ht_hash_id(uniq_id)];
    if( NULL == list_head->item ){
        return NULL;
    }

    for(curr=list_head; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This can only legally happen for the head of the list.
            SDE_ERROR("ht_lookup_by_id() the hash table is clobbered.");
            return NULL;
        }
        if(curr->item->glb_uniq_id == uniq_id){
            return curr->item;
        }
    }

    return NULL;
}


/******************************************************************************/
/* Functions related to the exponential container used for recorders.         */
/******************************************************************************/
void exp_container_to_contiguous(recorder_data_t *exp_container, void *cont_buffer){
    long long current_size, typesize, used_entries, tmp_size = 0;
    void *src, *dst;
    int i;

    typesize = exp_container->typesize;
    used_entries = exp_container->used_entries;

    for(i=0; i<EXP_CONTAINER_ENTRIES; i++){
       current_size = ((long long)1<<i) * EXP_CONTAINER_MIN_SIZE;
       src = exp_container->ptr_array[i];
       dst = (char *)cont_buffer + tmp_size*typesize;
       if ( (tmp_size+current_size) <= used_entries){
           memcpy(dst, src, current_size*typesize);
           if ( (tmp_size+current_size) == used_entries){
               return;
           }
       }else{
           memcpy(dst, src, (used_entries-tmp_size)*typesize);
           return;
       }
       tmp_size += current_size;
    }
}

int exp_container_insert_element(recorder_data_t *exp_container, size_t typesize, const void *value){
    long long used_entries, total_entries, prev_entries, offset;
    int i, chunk;
    long long tmp_size;

    if( NULL == exp_container || NULL == exp_container->ptr_array[0]){
        SDE_ERROR("exp_container_insert_element(): Exponential container is clobbered. Unable to insert element.");
        return SDE_EINVAL;
    }

    used_entries = exp_container->used_entries;
    total_entries = exp_container->total_entries;
    assert(used_entries <= total_entries);

    // Find how many chunks we have already allocated
    tmp_size = 0;
    for(i=0; i<EXP_CONTAINER_ENTRIES; i++){
       long long factor = (long long)1<<i; // 2^i;
       prev_entries = tmp_size;
       tmp_size += factor * EXP_CONTAINER_MIN_SIZE;
       // At least the first chunk "exp_container->ptr_array[0]"
       // must have been already allocated when creating the recorder, so we can
       // compare the total size after we add the "i-th" size.
       if (total_entries == tmp_size)
           break;
    }
    chunk = i;

    // Find how many entries down the last chunk we are.
    offset = used_entries - prev_entries;

    if( used_entries == total_entries ){
        long long new_segment_size;

        // If we had used all the available entries (and thus we are allocating more), we start from the beginning of the new chunk.
        offset = 0;

        chunk += 1; // we need to allocate the next chunk from the last one we found.
        new_segment_size = ((long long)1<<chunk) * EXP_CONTAINER_MIN_SIZE;
        exp_container->ptr_array[chunk] = malloc(new_segment_size*typesize);
        exp_container->total_entries += new_segment_size;
    }

    void *dest = (char *)(exp_container->ptr_array[chunk]) + offset*typesize;

    (void)memcpy( dest, value, typesize );
    exp_container->used_entries++;

    return SDE_OK;
}

/******************************************************************************/
/* Functions related to the F14 inspired hash-table that we used to implement */
/* the counting set.                                                          */
/******************************************************************************/

int cset_insert_elem(cset_hash_table_t *hash_ptr, size_t element_size, size_t hashable_size, const void *element, uint32_t type_id){
    cset_hash_bucket_t *bucket_ptr;
    int element_in_bucket = 0;
    uint32_t vacant_idx, i, occupied;
    int ret_val;

    if( NULL == hash_ptr ){
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }
    bucket_ptr = hash_ptr->buckets;

    uint64_t seed = (uint64_t)79365; // decided to be a good seed by a committee.
    uint64_t key = fasthash64(element, hashable_size, seed);
    int bucket_idx = (int)(key % _SDE_HASH_BUCKET_COUNT_);
    uint64_t *key_ptr = bucket_ptr[bucket_idx].keys;
    cset_hash_decorated_object_t *obj_ptr = bucket_ptr[bucket_idx].objects;
    occupied = bucket_ptr[bucket_idx].occupied;
    if( occupied > _SDE_HASH_BUCKET_WIDTH_ ){
        SDE_ERROR("cset_insert_elem(): Counting set is clobbered, bucket %d has exceeded capacity.",bucket_idx);
        ret_val = SDE_ECMP;
        goto fn_exit;
    }

    // First look in the bucket where the hash function told us to look.
    for(i=0; i<occupied; i++){
        // If the key and type_id match a stored element and the hashable_size is less or equal to
        // the size of the stored element, then we are onto something.
        if( (key == key_ptr[i]) && (type_id == obj_ptr[i].type_id) && (hashable_size <= obj_ptr[i].type_size) ){
            // If the actual element matches too (or if we don't care about perfect matches),
            // then we update the count for this entry and we are done.
            if( SDE_HASH_IS_FUZZY || !memcmp(element, obj_ptr[i].ptr, hashable_size) ){
                obj_ptr[i].count += 1;
                element_in_bucket = 1;
                break;
            }
        }
    }
    // After the loop "i" will contain the index of the first element that is not occupied.
    vacant_idx = i;

    // If we didn't find the element in the appropriate bucket, then we need to add it (or look in the overflow list).
    if( !element_in_bucket ){
        // If the overflow list is empty, then we are certainly dealing with a new element.
        if( NULL == hash_ptr->overflow_list ){
            // Check if we still have room in the bucket, and if so, add the new element to the bucket.
            if( vacant_idx < _SDE_HASH_BUCKET_WIDTH_ ){
                key_ptr[vacant_idx] = key;
                obj_ptr[vacant_idx].count = 1;
                obj_ptr[vacant_idx].type_id = type_id;
                obj_ptr[vacant_idx].type_size = element_size;
                obj_ptr[vacant_idx].ptr = malloc(element_size);
                (void)memcpy(obj_ptr[vacant_idx].ptr, element, element_size);
                // Let the bucket know that it now has one more element.
                bucket_ptr[bucket_idx].occupied += 1;
            }else{
                // If the overflow list is empty and the bucket does not have room,
                // then we add the new element at the head of the overflow list.
                cset_list_object_t *new_list_element = (cset_list_object_t *)malloc(sizeof(cset_list_object_t));
                new_list_element->next = NULL;
                new_list_element->count = 1;
                new_list_element->type_id = type_id;
                new_list_element->type_size = element_size;
                new_list_element->ptr = malloc(element_size);
                (void)memcpy(new_list_element->ptr, element, element_size);
                // Make the head point to the new element.
                hash_ptr->overflow_list = new_list_element;
            }
        }else{
            int element_in_list = 0;
            // Since there are elements in the overflow list, we need to search there for the one we are looking for.
            cset_list_object_t *list_runner;
            for( list_runner = hash_ptr->overflow_list; list_runner != NULL; list_runner = list_runner->next){
                // if we find the element in the overflow list, increment the counter and exit the loop.
                // When we traverse the overflow list we can _not_ use the SDE_HASH_IS_FUZZY flag, because we
                // don't have matching hashes to indicate that the two elements are close; we are traversing the whole list.
                if( (hashable_size <= list_runner->type_size) && (type_id == list_runner->type_id) && !memcmp(element, list_runner->ptr, hashable_size) ){
                    list_runner->count += 1;
                    element_in_list = 1;
                    break;
                }
            }
            // If we traversed the entire list and still haven't found our element, we need to insert it into the cset.
            if( !element_in_list ){
                // Check if there is still room in the bucket. If there is, we should add the new element to the bucket instead of the overflow list.
                if( vacant_idx < _SDE_HASH_BUCKET_WIDTH_ ){
                    key_ptr[vacant_idx] = key;
                    obj_ptr[vacant_idx].count = 1;
                    obj_ptr[vacant_idx].type_id = type_id;
                    obj_ptr[vacant_idx].type_size = element_size;
                    obj_ptr[vacant_idx].ptr = malloc(element_size);
                    (void)memcpy(obj_ptr[vacant_idx].ptr, element, element_size);
                    // Let the bucket know that it now has one more element.
                    bucket_ptr[bucket_idx].occupied += 1;
                }else{
                    // If there is no room in the bucket, add the new element before the current head of the overflow list.
                    cset_list_object_t *new_list_element = (cset_list_object_t *)malloc(sizeof(cset_list_object_t));
                    // Make the new element's "next" pointer be the current head of the list.
                    new_list_element->next = hash_ptr->overflow_list;
                    new_list_element->count = 1;
                    new_list_element->type_id = type_id;
                    new_list_element->type_size = element_size;
                    new_list_element->ptr = malloc(element_size);
                    (void)memcpy(new_list_element->ptr, element, element_size);
                    // Update the head of the list to point to the new element.
                    hash_ptr->overflow_list = new_list_element;
                }
            }
        }
    }

    ret_val = SDE_OK;
fn_exit:
    return ret_val;
}


int cset_remove_elem(cset_hash_table_t *hash_ptr, size_t hashable_size, const void *element, uint32_t type_id){
    cset_hash_bucket_t *bucket_ptr;
    int element_found = 0;
    uint32_t i, occupied;
    int ret_val;

    if( NULL == hash_ptr ){
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }
    bucket_ptr = hash_ptr->buckets;

    uint64_t seed = (uint64_t)79365; // decided to be a good seed by a committee.
    uint64_t key = fasthash64(element, hashable_size, seed);
    int bucket_idx = (int)(key % _SDE_HASH_BUCKET_COUNT_);
    uint64_t *key_ptr = bucket_ptr[bucket_idx].keys;
    cset_hash_decorated_object_t *obj_ptr = bucket_ptr[bucket_idx].objects;
    occupied = bucket_ptr[bucket_idx].occupied;
    if( occupied > _SDE_HASH_BUCKET_WIDTH_ ){
        SDE_ERROR("cset_remove_elem(): Counting set is clobbered, bucket %d has exceeded capacity.",bucket_idx);
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    // First look in the bucket where the hash function told us to look.
    for(i=0; i<occupied; i++){
        // If the key and type_id match a stored element and the hashable_size is less or equal to
        // the size of the stored element, then we are onto something.
        if( (key == key_ptr[i]) && (type_id == obj_ptr[i].type_id) && (hashable_size <= obj_ptr[i].type_size) ){
            // If the actual element matches too (or if we don't care about perfect matches),
            // then we update the count for this entry.
            if( SDE_HASH_IS_FUZZY || !memcmp(element, obj_ptr[i].ptr, hashable_size) ){
                obj_ptr[i].count -= 1;
                // If the element reached a count of zero after we removed it, then shift all the other keys and entries in the bucket.
                if( 0 == obj_ptr[i].count ){
                    uint32_t j;
                    // free the memory taken by the user object.
                    free(obj_ptr[i].ptr);
                    // now shift the remaining entries in this bucket.
                    for(j=i; j<occupied-1; j++){
                        key_ptr[j] = key_ptr[j+1];
                        obj_ptr[j] = obj_ptr[j+1];
                    }
                    bucket_ptr[bucket_idx].occupied -= 1;
                }
                // since we found the element, we don't need to look further.
                element_found = 1;
                break;
            }
        }
    }

    // If we didn't find the element in the appropriate bucket, then we need to look for it in the overflow list.
    if( !element_found ){
        // If the overflow list is empty, then something went wrong.
        if( NULL == hash_ptr->overflow_list ){
            SDE_ERROR("cset_remove_elem(): Attempted to remove element that is NOT in the counting set.");
        }else{
            // Since there are elements in the overflow list, we need to search there for the one we are looking for.
            cset_list_object_t *list_runner, *prev;
            prev = hash_ptr->overflow_list;
            for( list_runner = hash_ptr->overflow_list; list_runner != NULL; list_runner = list_runner->next){
                // if we find the element in the overflow list
                if( (hashable_size <= list_runner->type_size) && (type_id == list_runner->type_id) && !memcmp(element, list_runner->ptr, hashable_size) ){
                    list_runner->count -= 1;
                    // If the element reached a count of zero, then remove it from the list, and connect the list around it.
                    if( 0 == list_runner->count ){
                        // free the memory taken by the user object.
                        free(list_runner->ptr);
                        if( list_runner == hash_ptr->overflow_list ){
                            hash_ptr->overflow_list = NULL;
                        }else{
                            prev->next = list_runner->next;
                        }
                        // free the memory taken by the link node. We can do this here safely
                        // because we will break out of the loop, so we will not need the "next" pointer.
                        free(list_runner);
                    }
                    // since we found the element, we don't need to look at the rest of the list.
                    break;
                }
                prev = list_runner;
            }
        }
    }

    ret_val = SDE_OK;
fn_exit:
    return ret_val;
}

cset_list_object_t *cset_to_list(cset_hash_table_t *hash_ptr){
    cset_hash_bucket_t *bucket_ptr;
    int bucket_idx;
    uint32_t i, occupied;
    cset_list_object_t *head_ptr = NULL;

    if( NULL == hash_ptr ){
        return NULL;
    }
    bucket_ptr = hash_ptr->buckets;

    for( bucket_idx = 0; bucket_idx < _SDE_HASH_BUCKET_COUNT_; bucket_idx++){
        cset_hash_decorated_object_t *obj_ptr = bucket_ptr[bucket_idx].objects;
        occupied = bucket_ptr[bucket_idx].occupied;

        for(i=0; i<occupied; i++){
            int type_size = obj_ptr[i].type_size;
            cset_list_object_t *new_list_element = (cset_list_object_t *)malloc(sizeof(cset_list_object_t));
            // make the current list head be the element after the new one we are creating.
            new_list_element->next = head_ptr;
            new_list_element->count = obj_ptr[i].count;
            new_list_element->type_id = obj_ptr[i].type_id;
            new_list_element->type_size = type_size;
            new_list_element->ptr = malloc(type_size);
            (void)memcpy(new_list_element->ptr, obj_ptr[i].ptr, type_size);
            // Update the head of the list to point to the new element.
            head_ptr = new_list_element;
        }
     }

    cset_list_object_t *list_runner;
    // Since there are elements in the overflow list, we need to search for ours.
    for( list_runner = hash_ptr->overflow_list; list_runner != NULL; list_runner = list_runner->next){
        int type_size = list_runner->type_size;
        cset_list_object_t *new_list_element = (cset_list_object_t *)malloc(sizeof(cset_list_object_t));
        // make the current list head be the element after the new one we are creating.
        new_list_element->next = head_ptr;
        new_list_element->count = list_runner->count;
        new_list_element->type_id = list_runner->type_id;
        new_list_element->type_size = type_size;
        new_list_element->ptr = malloc(type_size);
        (void)memcpy(new_list_element->ptr, list_runner->ptr, type_size);
        // Update the head of the list to point to the new element.
        head_ptr = new_list_element;
    }

    return head_ptr;
}


int cset_delete(cset_hash_table_t *hash_ptr){
    cset_hash_bucket_t *bucket_ptr;
    int bucket_idx;
    uint32_t i, occupied;

    if( NULL == hash_ptr ){
        return SDE_EINVAL;
    }
    bucket_ptr = hash_ptr->buckets;

    for( bucket_idx = 0; bucket_idx < _SDE_HASH_BUCKET_COUNT_; bucket_idx++){
        cset_hash_decorated_object_t *obj_ptr = bucket_ptr[bucket_idx].objects;
        occupied = bucket_ptr[bucket_idx].occupied;
        // Free all the elements that occupy entries in this bucket.
        for(i=0; i<occupied; i++){
            free(obj_ptr[i].ptr);
        }
        bucket_ptr[bucket_idx].occupied = 0;
     }

    cset_list_object_t *list_runner, *ptr_to_free=NULL;
    // Since there are elements in the overflow list, we need to search for ours.
    for( list_runner = hash_ptr->overflow_list; list_runner != NULL; list_runner = list_runner->next){
        // Free the list element from the previous iteration.
        free(ptr_to_free);
        free(list_runner->ptr);
        // Keep a reference to this element so we can free it _after_ this iteration, because we need the list_runner->next for now.
        ptr_to_free = list_runner;
        // If the current element is at the head of the overflow list, then we should mark the head as NULL.
        if( list_runner == hash_ptr->overflow_list )
            hash_ptr->overflow_list = NULL;

    }
    free(ptr_to_free);

    return SDE_OK;
}

