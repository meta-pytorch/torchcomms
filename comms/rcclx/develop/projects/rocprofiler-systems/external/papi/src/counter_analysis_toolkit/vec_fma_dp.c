#include "vec_scalar_verify.h"

static double test_dp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
static double test_dp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
static double test_dp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
static void   test_dp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_dp_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_dp_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_dp_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(ARM)
void test_dp_arm_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(POWER)
void test_dp_power_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_dp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#endif

/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
static
double test_dp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp ){
    register DP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PD(0.01);
    r1 = SET_VEC_PD(0.02);
    r2 = SET_VEC_PD(0.03);
    r3 = SET_VEC_PD(0.04);
    r4 = SET_VEC_PD(0.05);
    r5 = SET_VEC_PD(0.06);
    r6 = SET_VEC_PD(0.07);
    r7 = SET_VEC_PD(0.08);
    r8 = SET_VEC_PD(0.09);
    r9 = SET_VEC_PD(0.10);
    rA = SET_VEC_PD(0.11);
    rB = SET_VEC_PD(0.12);
    rC = SET_VEC_PD(0.13);
    rD = SET_VEC_PD(0.14);
    rE = SET_VEC_PD(0.15);
    rF = SET_VEC_PD(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(12, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PD(r0,r1);
    r2 = ADD_VEC_PD(r2,r3);
    r4 = ADD_VEC_PD(r4,r5);

    r0 = ADD_VEC_PD(r0,r6);
    r2 = ADD_VEC_PD(r2,r4);

    r0 = ADD_VEC_PD(r0,r2);

    double out = 0;
    DP_VEC_TYPE temp = r0;
    out += ((double*)&temp)[0];
    out += ((double*)&temp)[1];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
static
double test_dp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp ){
    register DP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PD(0.01);
    r1 = SET_VEC_PD(0.02);
    r2 = SET_VEC_PD(0.03);
    r3 = SET_VEC_PD(0.04);
    r4 = SET_VEC_PD(0.05);
    r5 = SET_VEC_PD(0.06);
    r6 = SET_VEC_PD(0.07);
    r7 = SET_VEC_PD(0.08);
    r8 = SET_VEC_PD(0.09);
    r9 = SET_VEC_PD(0.10);
    rA = SET_VEC_PD(0.11);
    rB = SET_VEC_PD(0.12);
    rC = SET_VEC_PD(0.13);
    rD = SET_VEC_PD(0.14);
    rE = SET_VEC_PD(0.15);
    rF = SET_VEC_PD(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(24, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PD(r0,r1);
    r2 = ADD_VEC_PD(r2,r3);
    r4 = ADD_VEC_PD(r4,r5);

    r0 = ADD_VEC_PD(r0,r6);
    r2 = ADD_VEC_PD(r2,r4);

    r0 = ADD_VEC_PD(r0,r2);

    double out = 0;
    DP_VEC_TYPE temp = r0;
    out += ((double*)&temp)[0];
    out += ((double*)&temp)[1];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
static
double test_dp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp ){
    register DP_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PD(0.01);
    r1 = SET_VEC_PD(0.02);
    r2 = SET_VEC_PD(0.03);
    r3 = SET_VEC_PD(0.04);
    r4 = SET_VEC_PD(0.05);
    r5 = SET_VEC_PD(0.06);
    r6 = SET_VEC_PD(0.07);
    r7 = SET_VEC_PD(0.08);
    r8 = SET_VEC_PD(0.09);
    r9 = SET_VEC_PD(0.10);
    rA = SET_VEC_PD(0.11);
    rB = SET_VEC_PD(0.12);
    rC = SET_VEC_PD(0.13);
    rD = SET_VEC_PD(0.14);
    rE = SET_VEC_PD(0.15);
    rF = SET_VEC_PD(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            r0 = FMA_VEC_PD(r0,r7,r9);
            r1 = FMA_VEC_PD(r1,r8,rA);
            r2 = FMA_VEC_PD(r2,r9,rB);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,rB,rD);
            r5 = FMA_VEC_PD(r5,rC,rE);

            r0 = FMA_VEC_PD(r0,rD,rF);
            r1 = FMA_VEC_PD(r1,rC,rE);
            r2 = FMA_VEC_PD(r2,rB,rD);
            r3 = FMA_VEC_PD(r3,rA,rC);
            r4 = FMA_VEC_PD(r4,r9,rB);
            r5 = FMA_VEC_PD(r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(48, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PD(r0,r1);
    r2 = ADD_VEC_PD(r2,r3);
    r4 = ADD_VEC_PD(r4,r5);

    r0 = ADD_VEC_PD(r0,r6);
    r2 = ADD_VEC_PD(r2,r4);

    r0 = ADD_VEC_PD(r0,r2);

    double out = 0;
    DP_VEC_TYPE temp = r0;
    out += ((double*)&temp)[0];
    out += ((double*)&temp)[1];

    return out;
}

static
void test_dp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    double sum = 0.0;
    double scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_dp_mac_VEC_FMA_12( iterations, EventSet, fp );
        scalar_sum += test_dp_scalar_VEC_FMA_12( iterations );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_dp_mac_VEC_FMA_24( iterations, EventSet, fp );
        scalar_sum += test_dp_scalar_VEC_FMA_24( iterations );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_dp_mac_VEC_FMA_48( iterations, EventSet, fp );
        scalar_sum += test_dp_scalar_VEC_FMA_48( iterations );
    }

    if( sum/2.0 != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}
