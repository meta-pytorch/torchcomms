#include <stdio.h>
#include <stdlib.h>
#include "sde_lib.h"
#include "papi.h"
#include "papi_test.h"

papi_handle_t handle;

typedef struct test_type_s{
    int id;
    float x;
    double y;
} test_type_t;

typedef struct mem_type_s{
    void *ptr;
    int line_of_code;
    size_t size;
} mem_type_t;

void libCSet_do_simple_work(void){
    int i;
    void *test_set;
    test_type_t element;

    handle = papi_sde_init("CSET_LIB");
    papi_sde_create_counting_set( handle, "test counting set", &test_set );

    for(i=0; i<22390; i++){ 
        int j = i%5222;
        long r = random();
        if ( r%23 == 0 ){
            j = r%2287;
        }
        element.id = j;
        element.x = (float)j*1.037/((float)j+32.1);
        element.y = (double)(element.x)+145.67/((double)j+0.01);
        papi_sde_counting_set_insert( test_set, sizeof(element), sizeof(element), &element, 0);
    }

    return;
}

int libCSet_finalize(void){
    return papi_sde_shutdown(handle);
}

void libCSet_do_memory_allocations(void){
    int i, iter;
    void *mem_set;
    void *ptrs[128];

    handle = papi_sde_init("CSET_LIB");
    papi_sde_create_counting_set( handle, "malloc_tracking", &mem_set );

    for(iter=0; iter<8; iter++){
        mem_type_t alloc_elem;

        for(i=0; i<64; i++){
            size_t len = (17+i)*73;
            ptrs[i] = malloc(len);

            alloc_elem.ptr = ptrs[i];
            alloc_elem.line_of_code = __LINE__;
            alloc_elem.size = len;
            papi_sde_counting_set_insert( mem_set, sizeof(alloc_elem), sizeof(void *), &alloc_elem, 1);
        }
        // "i" does _not_ start from zero so that some pointers are _not_ free()ed
        for(i=iter; i<64; i++){
            papi_sde_counting_set_remove( mem_set, sizeof(void *), &(ptrs[i]), 1);
            free(ptrs[i]);
        }

        for(i=0; i<32; i++){
            size_t len = (19+i)*73;
            ptrs[i] = malloc(len);

            alloc_elem.ptr = ptrs[i];
            alloc_elem.line_of_code = __LINE__;
            alloc_elem.size = len;
            papi_sde_counting_set_insert( mem_set, sizeof(alloc_elem), sizeof(void *), &alloc_elem, 1);
        }
        // "i" does _not_ go to 31 so that some pointers are _not_ free()ed
        for(i=0; i<32-iter; i++){
            papi_sde_counting_set_remove( mem_set, sizeof(void *), &(ptrs[i]), 1);
            free(ptrs[i]);
        }

    }

    return;
}

void libCSet_dump_set( cset_list_object_t *list_head ){
    cset_list_object_t *list_runner;

    for(list_runner = list_head; NULL != list_runner; list_runner=list_runner->next){

        switch(list_runner->type_id){
            case 0:
                {
                test_type_t *ptr = (test_type_t *)(list_runner->ptr);
                printf("count= %d typesize= %lu {id= %d, x= %f, y= %lf}\n", list_runner->count, list_runner->type_size, ptr->id, ptr->x, ptr->y);
                break;
                }
            case 1:
                {
                mem_type_t *ptr = (mem_type_t *)(list_runner->ptr);
                printf("count= %d typesize= %lu { ptr= %p, line= %d, size= %lu }\n", list_runner->count, list_runner->type_size, ptr->ptr, ptr->line_of_code, ptr->size);
                break;
                }
        }
    }

    return;
}

int libCSet_count_set_elements( cset_list_object_t *list_head ){
    cset_list_object_t *list_runner;
    int element_count = 0;

    for(list_runner = list_head; NULL != list_runner; list_runner=list_runner->next){
        ++element_count;
    }

    return element_count;
}


// Hook for papi_native_avail utility. No user code which links against this library should call
// this function because it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    papi_handle_t tmp_handle;
    tmp_handle = fptr_struct->init("CSET_LIB");
    fptr_struct->create_counting_set( tmp_handle, "test counting set", NULL );
    fptr_struct->create_counting_set( tmp_handle, "malloc_tracking", NULL );
    return tmp_handle;
}
