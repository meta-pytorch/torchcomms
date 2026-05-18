#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include "sde_lib.h"
#include "sde_lib.hpp"

static const char *event_names[1] = {
    "simple_recording"
};

#define BRNG() {\
    b  = ((z1 << 6) ^ z1) >> 13;\
    z1 = ((z1 & 4294967294U) << 18) ^ b;\
    b  = ((z2 << 2) ^ z2) >> 27;\
    z2 = ((z2 & 4294967288U) << 2) ^ b;\
    b  = ((z3 << 13) ^ z3) >> 21;\
    z3 = ((z3 & 4294967280U) << 7) ^ b;\
    b  = ((z4 << 3) ^ z4) >> 12;\
    z4 = ((z4 & 4294967168U) << 13) ^ b;\
    z1++;\
    result = z1 ^ z2 ^ z3 ^ z4;\
}
volatile int result;
volatile unsigned int b, z1, z2, z3, z4;

void *rcrd_handle;
papi_sde::PapiSde::Recorder *sde_rcrd;

// API functions.
void recorder_init_(void);
void recorder_do_work_(void);

void recorder_init_(void){
    papi_sde::PapiSde sde("CPP_Lib_With_Recorder");
    sde_rcrd = sde.create_recorder(event_names[0], sizeof(long long), papi_sde_compare_long_long);
    if( nullptr == sde_rcrd ){
        std::cerr << "Unable to create recorder: "<< event_names[0] << std::endl;
        abort();
    }

    z1=42;
    z2=420;
    z3=42000;
    z4=424242;

    return;
}

void recorder_do_work_(void){
    long long r;
    BRNG();
    if( result < 0 )
        result *= -1;
    r = result%123456;
    sde_rcrd->record(r);
    return;
}

// Hook for papi_native_avail utility. No user code which links against this library should call
// this function because it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    papi_handle_t tmp_handle;
    tmp_handle = fptr_struct->init("CPP_Lib_With_Recorder");
    fptr_struct->create_recorder(tmp_handle, event_names[0], sizeof(long long), papi_sde_compare_long_long, &rcrd_handle);
    return tmp_handle;
}
