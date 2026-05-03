#include <stdio.h>
#include "vec.h"

#pragma weak vec_driver

void __attribute__((weak)) vec_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    (void)hw_desc;

    fprintf(stderr, "Failed to create %s.vec in %s. The Vector FLOP benchmark is not supported on this architecture!\n", papi_event_name, outdir);
}
