#include <stdio.h>
#include <cstdlib>
#include "sde_lib.h"
#include "papi.h"
#include "papi_test.h"
#include "cset_lib.hpp"

struct test_type_t{
    int id;
    float x;
    double y;
};

struct mem_type_t{
    void *ptr;
    int line_of_code;
    size_t size;
};

CSetLib::CSetLib(){
    papi_sde::PapiSde sde("CPP_CSET_LIB");
    test_set = sde.create_counting_set("test counting set");
    mem_set  = sde.create_counting_set("malloc_tracking");
}

void CSetLib::do_simple_work(){
    int i;
    test_type_t element;

    for(i=0; i<22390; i++){ 
        int j = i%5222;
        element.id = j;
        element.x = (float)j*1.037/((float)j+32.1);
        element.y = (double)(element.x)+145.67/((double)j+0.01);
        test_set->insert(sizeof(element), element, 0);
    }

    return;
}

void CSetLib::do_memory_allocations(){
    int i, iter;
    void *ptrs[128];
    
    for(iter=0; iter<8; iter++){
        mem_type_t alloc_elem;

        for(i=0; i<64; i++){
            size_t len = (17+i)*73;
            ptrs[i] = malloc(len);

            alloc_elem.ptr = ptrs[i];
            alloc_elem.line_of_code = __LINE__;
            alloc_elem.size = len;
            mem_set->insert(sizeof(void *), alloc_elem, 1);
        }
        for(i=iter; i<64; i++){
            mem_set->remove(sizeof(void *), ptrs[i], 1);
            free(ptrs[i]);
        }

        for(i=0; i<32; i++){
            size_t len = (19+i)*73;
            ptrs[i] = malloc(len);

            alloc_elem.ptr = ptrs[i];
            alloc_elem.line_of_code = __LINE__;
            alloc_elem.size = len;
            mem_set->insert(sizeof(void *), alloc_elem, 1);
        }
        for(i=0; i<32-iter; i++){
            mem_set->remove(sizeof(void *), ptrs[i], 1);
            free(ptrs[i]);
        }

    }
}

void CSetLib::dump_set( cset_list_object_t *list_head ){
    cset_list_object_t *list_runner;

    for(list_runner = list_head; NULL != list_runner; list_runner=list_runner->next){
        switch(list_runner->type_id){
            case 0:
                {
                auto *ptr = static_cast<test_type_t *>(list_runner->ptr);
                printf("count= %d typesize= %lu {id= %d, x= %f, y= %lf}\n", list_runner->count, list_runner->type_size, ptr->id, ptr->x, ptr->y);
                break;
                }
            case 1:
                {
                auto *ptr = static_cast<mem_type_t *>(list_runner->ptr);
                printf("count= %d typesize= %lu { ptr= %p, line= %d, size= %lu }\n", list_runner->count, list_runner->type_size, ptr->ptr, ptr->line_of_code, ptr->size);
                break;
                }
        }
    }
}


int CSetLib::count_set_elements( cset_list_object_t *list_head ){
    cset_list_object_t *list_runner;
    int element_counter = 0;

    for(list_runner = list_head; NULL != list_runner; list_runner=list_runner->next){
        ++element_counter;
    }

    return element_counter;
}

// Hook for papi_native_avail utility. No user code which links against this library should call
// this function because it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
extern "C" papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    papi_handle_t tmp_handle;
    tmp_handle = fptr_struct->init("CPP_CSET_LIB");
    fptr_struct->create_counting_set( tmp_handle, "test counting set", NULL );
    fptr_struct->create_counting_set( tmp_handle, "malloc_tracking", NULL );
    return tmp_handle;
}
