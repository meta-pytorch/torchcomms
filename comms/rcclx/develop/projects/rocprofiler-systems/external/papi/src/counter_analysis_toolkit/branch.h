#ifndef _BRANCH_
#define _BRANCH_

#include "hw_desc.h"

#define BRANCH_BENCH(_I_) {\
    iter = 0;\
    avg = 0.0;\
    for(i=512; i<max_iter; i*=2){\
        iter++;\
        sz = i;\
        cnt = branch_char_b ## _I_ (sz, papi_eventset);\
        avg += (double)cnt/(double)sz;\
        sz = (int)((double)i*1.1892);\
        cnt = branch_char_b ## _I_ (sz, papi_eventset);\
        avg += (double)cnt/(double)sz;\
        sz = (int)((double)i*1.4142);\
        cnt = branch_char_b ## _I_ (sz, papi_eventset);\
        avg += (double)cnt/(double)sz;\
        sz = (int)((double)i*1.6818);\
        cnt = branch_char_b ## _I_ (sz, papi_eventset);\
        avg += (double)cnt/(double)sz;\
    }\
    if(avg < 0.0){\
        fclose(ofp_papi);\
        return;\
    }\
    avg = avg/(4.0*(double)iter);\
    round = floor(avg*4.0+0.499)/4.0;\
    fprintf(ofp_papi,"%.2lf\n", round);\
}

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

#define BUSY_WORK() {BRNG(); BRNG(); BRNG(); BRNG();}

extern volatile int result;
extern volatile unsigned int b, z1, z2, z3, z4;

void branch_driver(char *papi_event_name, int junk, hw_desc_t *hw_desc, char* outdir);
long long int branch_char_b1(int size, int papi_eventset);
long long int branch_char_b2(int size, int papi_eventset);
long long int branch_char_b3(int size, int papi_eventset);
long long int branch_char_b4(int size, int papi_eventset);
long long int branch_char_b4a(int size, int papi_eventset);
long long int branch_char_b4b(int size, int papi_eventset);
long long int branch_char_b5(int size, int papi_eventset);
long long int branch_char_b5a(int size, int papi_eventset);
long long int branch_char_b5b(int size, int papi_eventset);
long long int branch_char_b6(int size, int papi_eventset);
long long int branch_char_b7(int size, int papi_eventset);

#endif
