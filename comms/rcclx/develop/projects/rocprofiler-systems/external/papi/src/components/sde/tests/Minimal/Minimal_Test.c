#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"
#include "papi_test.h"
#include "sde_lib.h"

long long local_var;

void mintest_init(void){
    local_var =0;
    papi_handle_t handle = papi_sde_init("Min Example Code");
    papi_sde_register_counter(handle, "Example Event", PAPI_SDE_RO|PAPI_SDE_DELTA, PAPI_SDE_long_long, &local_var);
}

void mintest_dowork(void){
    local_var += 7;
}

int main(int argc, char **argv){
    int ret, Eventset = PAPI_NULL;
    long long counter_values[1];

    (void)argc;
    (void)argv;

    mintest_init();

    // --- Setup PAPI
    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        test_fail( __FILE__, __LINE__, "PAPI_library_init", ret );
        exit(-1);
    }

    if((ret=PAPI_create_eventset(&Eventset)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );
        exit(-1);
    }

    if((ret=PAPI_add_named_event(Eventset, "sde:::Min Example Code::Example Event")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
        exit(-1);
    }

    // --- Start PAPI
    if((ret=PAPI_start(Eventset)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_start", ret );
        exit(-1);
    }

    mintest_dowork();

    // --- Stop PAPI
    if((ret=PAPI_stop(Eventset, counter_values)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
        exit(-1);
    }

    if( counter_values[0] == 7 ){
        test_pass(__FILE__);
    }else{
        test_fail( __FILE__, __LINE__, "SDE counter values are wrong!", ret );
    }

    return 0;
}
