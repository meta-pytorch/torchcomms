#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"
#include "papi_test.h"
#include "sde_lib.h"
#include "sde_lib.hpp"


////////////////////////////////////////////////////////////////////////////////
//------- Library example that exports SDEs

class MinTest{
public:
    MinTest();
    void dowork();

private:
    long long local_var;
};

MinTest::MinTest(){
    local_var = 0;
    papi_sde::PapiSde sde("Min Example Code in C++");
    sde.register_counter("Example Event", PAPI_SDE_RO|PAPI_SDE_DELTA, local_var);
}

void MinTest::dowork(){
    local_var += 7;
}

////////////////////////////////////////////////////////////////////////////////
//------- Driver example that uses library and reads the SDEs

int main(int argc, char **argv){
    int ret, Eventset = PAPI_NULL;
    long long counter_values[1];
    MinTest test_obj;

    (void)argc;
    (void)argv;

    // --- Setup PAPI
    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        test_fail( __FILE__, __LINE__, "PAPI_library_init", ret );
        exit(-1);
    }

    if((ret=PAPI_create_eventset(&Eventset)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );
        exit(-1);
    }

    if((ret=PAPI_add_named_event(Eventset, "sde:::Min Example Code in C++::Example Event")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
        exit(-1);
    }

    // --- Start PAPI
    if((ret=PAPI_start(Eventset)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_start", ret );
        exit(-1);
    }

    test_obj.dowork();

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
