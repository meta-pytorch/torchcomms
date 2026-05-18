#if !defined(CSET_LIB_H)
#define CSET_LIB_H

#include "sde_lib.h"
#include "sde_lib.hpp"

class CSetLib{
    private:
        papi_sde::PapiSde::CountingSet *test_set;
        papi_sde::PapiSde::CountingSet *mem_set;

	public:
        CSetLib();
        void do_simple_work();
        void do_memory_allocations();
        int count_set_elements( cset_list_object_t *list_head );
        void dump_set( cset_list_object_t *list_head );
};

#endif
