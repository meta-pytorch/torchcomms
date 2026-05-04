#include "vec_scalar_verify.h"

void papi_stop_and_print_placeholder(long long theory, FILE *fp)
{
    fprintf(fp, "%lld 0\n", theory);
}

void papi_stop_and_print(long long theory, int EventSet, FILE *fp)
{
    long long flpins = 0;
    int retval;

    if ( (retval=PAPI_stop(EventSet, &flpins)) != PAPI_OK){
        fprintf(stderr, "Problem.\n");
        return;
    }

    fprintf(fp, "%lld %lld\n", theory, flpins);
}

#if defined(ARM)
half test_hp_scalar_VEC_24( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SH(0.01);
    r1 = SET_VEC_SH(0.02);
    r2 = SET_VEC_SH(0.03);
    r3 = SET_VEC_SH(0.04);
    r4 = SET_VEC_SH(0.05);
    r5 = SET_VEC_SH(0.06);
    r6 = SET_VEC_SH(0.07);
    r7 = SET_VEC_SH(0.08);
    r8 = SET_VEC_SH(0.09);
    r9 = SET_VEC_SH(0.10);
    rA = SET_VEC_SH(0.11);
    rB = SET_VEC_SH(0.12);
    rC = SET_VEC_SH(0.13);
    rD = SET_VEC_SH(0.14);
    rE = SET_VEC_SH(0.15);
    rF = SET_VEC_SH(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SH(r0,r1);
    r2 = ADD_VEC_SH(r2,r3);
    r4 = ADD_VEC_SH(r4,r5);
    r6 = ADD_VEC_SH(r6,r7);
    r8 = ADD_VEC_SH(r8,r9);
    rA = ADD_VEC_SH(rA,rB);

    r0 = ADD_VEC_SH(r0,r2);
    r4 = ADD_VEC_SH(r4,r6);
    r8 = ADD_VEC_SH(r8,rA);

    r0 = ADD_VEC_SH(r0,r4);
    r0 = ADD_VEC_SH(r0,r8);

    half out = 0;
    half temp = r0;
    out = ADD_VEC_SH(out,temp);

    return out;
}

half test_hp_scalar_VEC_48( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SH(0.01);
    r1 = SET_VEC_SH(0.02);
    r2 = SET_VEC_SH(0.03);
    r3 = SET_VEC_SH(0.04);
    r4 = SET_VEC_SH(0.05);
    r5 = SET_VEC_SH(0.06);
    r6 = SET_VEC_SH(0.07);
    r7 = SET_VEC_SH(0.08);
    r8 = SET_VEC_SH(0.09);
    r9 = SET_VEC_SH(0.10);
    rA = SET_VEC_SH(0.11);
    rB = SET_VEC_SH(0.12);
    rC = SET_VEC_SH(0.13);
    rD = SET_VEC_SH(0.14);
    rE = SET_VEC_SH(0.15);
    rF = SET_VEC_SH(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SH(r0,r1);
    r2 = ADD_VEC_SH(r2,r3);
    r4 = ADD_VEC_SH(r4,r5);
    r6 = ADD_VEC_SH(r6,r7);
    r8 = ADD_VEC_SH(r8,r9);
    rA = ADD_VEC_SH(rA,rB);

    r0 = ADD_VEC_SH(r0,r2);
    r4 = ADD_VEC_SH(r4,r6);
    r8 = ADD_VEC_SH(r8,rA);

    r0 = ADD_VEC_SH(r0,r4);
    r0 = ADD_VEC_SH(r0,r8);

    half out = 0;
    half temp = r0;
    out = ADD_VEC_SH(out,temp);

    return out;
}

half test_hp_scalar_VEC_96( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SH(0.01);
    r1 = SET_VEC_SH(0.02);
    r2 = SET_VEC_SH(0.03);
    r3 = SET_VEC_SH(0.04);
    r4 = SET_VEC_SH(0.05);
    r5 = SET_VEC_SH(0.06);
    r6 = SET_VEC_SH(0.07);
    r7 = SET_VEC_SH(0.08);
    r8 = SET_VEC_SH(0.09);
    r9 = SET_VEC_SH(0.10);
    rA = SET_VEC_SH(0.11);
    rB = SET_VEC_SH(0.12);
    rC = SET_VEC_SH(0.13);
    rD = SET_VEC_SH(0.14);
    rE = SET_VEC_SH(0.15);
    rF = SET_VEC_SH(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            r0 = MUL_VEC_SH(r0,rC);
            r1 = ADD_VEC_SH(r1,rD);
            r2 = MUL_VEC_SH(r2,rE);
            r3 = ADD_VEC_SH(r3,rF);
            r4 = MUL_VEC_SH(r4,rC);
            r5 = ADD_VEC_SH(r5,rD);
            r6 = MUL_VEC_SH(r6,rE);
            r7 = ADD_VEC_SH(r7,rF);
            r8 = MUL_VEC_SH(r8,rC);
            r9 = ADD_VEC_SH(r9,rD);
            rA = MUL_VEC_SH(rA,rE);
            rB = ADD_VEC_SH(rB,rF);

            r0 = ADD_VEC_SH(r0,rF);
            r1 = MUL_VEC_SH(r1,rE);
            r2 = ADD_VEC_SH(r2,rD);
            r3 = MUL_VEC_SH(r3,rC);
            r4 = ADD_VEC_SH(r4,rF);
            r5 = MUL_VEC_SH(r5,rE);
            r6 = ADD_VEC_SH(r6,rD);
            r7 = MUL_VEC_SH(r7,rC);
            r8 = ADD_VEC_SH(r8,rF);
            r9 = MUL_VEC_SH(r9,rE);
            rA = ADD_VEC_SH(rA,rD);
            rB = MUL_VEC_SH(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SH(r0,r1);
    r2 = ADD_VEC_SH(r2,r3);
    r4 = ADD_VEC_SH(r4,r5);
    r6 = ADD_VEC_SH(r6,r7);
    r8 = ADD_VEC_SH(r8,r9);
    rA = ADD_VEC_SH(rA,rB);

    r0 = ADD_VEC_SH(r0,r2);
    r4 = ADD_VEC_SH(r4,r6);
    r8 = ADD_VEC_SH(r8,rA);

    r0 = ADD_VEC_SH(r0,r4);
    r0 = ADD_VEC_SH(r0,r8);

    half out = 0;
    half temp = r0;
    out = ADD_VEC_SH(out,temp);

    return out;
}

#else
float test_hp_scalar_VEC_24( uint64 iterations ){

    (void)iterations;
    return 0.0;
}

float test_hp_scalar_VEC_48( uint64 iterations ){

    (void)iterations;
    return 0.0;
}

float test_hp_scalar_VEC_96( uint64 iterations ){

    (void)iterations;
    return 0.0;
}
#endif

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_scalar_VEC_24( uint64 iterations ){
    register SP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SS(0.01);
    r1 = SET_VEC_SS(0.02);
    r2 = SET_VEC_SS(0.03);
    r3 = SET_VEC_SS(0.04);
    r4 = SET_VEC_SS(0.05);
    r5 = SET_VEC_SS(0.06);
    r6 = SET_VEC_SS(0.07);
    r7 = SET_VEC_SS(0.08);
    r8 = SET_VEC_SS(0.09);
    r9 = SET_VEC_SS(0.10);
    rA = SET_VEC_SS(0.11);
    rB = SET_VEC_SS(0.12);
    rC = SET_VEC_SS(0.13);
    rD = SET_VEC_SS(0.14);
    rE = SET_VEC_SS(0.15);
    rF = SET_VEC_SS(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SS(r0,r1);
    r2 = ADD_VEC_SS(r2,r3);
    r4 = ADD_VEC_SS(r4,r5);
    r6 = ADD_VEC_SS(r6,r7);
    r8 = ADD_VEC_SS(r8,r9);
    rA = ADD_VEC_SS(rA,rB);

    r0 = ADD_VEC_SS(r0,r2);
    r4 = ADD_VEC_SS(r4,r6);
    r8 = ADD_VEC_SS(r8,rA);

    r0 = ADD_VEC_SS(r0,r4);
    r0 = ADD_VEC_SS(r0,r8);

    float out = 0;
    SP_SCALAR_TYPE temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
float test_sp_scalar_VEC_48( uint64 iterations ){
    register SP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SS(0.01);
    r1 = SET_VEC_SS(0.02);
    r2 = SET_VEC_SS(0.03);
    r3 = SET_VEC_SS(0.04);
    r4 = SET_VEC_SS(0.05);
    r5 = SET_VEC_SS(0.06);
    r6 = SET_VEC_SS(0.07);
    r7 = SET_VEC_SS(0.08);
    r8 = SET_VEC_SS(0.09);
    r9 = SET_VEC_SS(0.10);
    rA = SET_VEC_SS(0.11);
    rB = SET_VEC_SS(0.12);
    rC = SET_VEC_SS(0.13);
    rD = SET_VEC_SS(0.14);
    rE = SET_VEC_SS(0.15);
    rF = SET_VEC_SS(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SS(r0,r1);
    r2 = ADD_VEC_SS(r2,r3);
    r4 = ADD_VEC_SS(r4,r5);
    r6 = ADD_VEC_SS(r6,r7);
    r8 = ADD_VEC_SS(r8,r9);
    rA = ADD_VEC_SS(rA,rB);

    r0 = ADD_VEC_SS(r0,r2);
    r4 = ADD_VEC_SS(r4,r6);
    r8 = ADD_VEC_SS(r8,rA);

    r0 = ADD_VEC_SS(r0,r4);
    r0 = ADD_VEC_SS(r0,r8);

    float out = 0;
    SP_SCALAR_TYPE temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
float test_sp_scalar_VEC_96( uint64 iterations ){
    register SP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SS(0.01);
    r1 = SET_VEC_SS(0.02);
    r2 = SET_VEC_SS(0.03);
    r3 = SET_VEC_SS(0.04);
    r4 = SET_VEC_SS(0.05);
    r5 = SET_VEC_SS(0.06);
    r6 = SET_VEC_SS(0.07);
    r7 = SET_VEC_SS(0.08);
    r8 = SET_VEC_SS(0.09);
    r9 = SET_VEC_SS(0.10);
    rA = SET_VEC_SS(0.11);
    rB = SET_VEC_SS(0.12);
    rC = SET_VEC_SS(0.13);
    rD = SET_VEC_SS(0.14);
    rE = SET_VEC_SS(0.15);
    rF = SET_VEC_SS(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            r0 = MUL_VEC_SS(r0,rC);
            r1 = ADD_VEC_SS(r1,rD);
            r2 = MUL_VEC_SS(r2,rE);
            r3 = ADD_VEC_SS(r3,rF);
            r4 = MUL_VEC_SS(r4,rC);
            r5 = ADD_VEC_SS(r5,rD);
            r6 = MUL_VEC_SS(r6,rE);
            r7 = ADD_VEC_SS(r7,rF);
            r8 = MUL_VEC_SS(r8,rC);
            r9 = ADD_VEC_SS(r9,rD);
            rA = MUL_VEC_SS(rA,rE);
            rB = ADD_VEC_SS(rB,rF);

            r0 = ADD_VEC_SS(r0,rF);
            r1 = MUL_VEC_SS(r1,rE);
            r2 = ADD_VEC_SS(r2,rD);
            r3 = MUL_VEC_SS(r3,rC);
            r4 = ADD_VEC_SS(r4,rF);
            r5 = MUL_VEC_SS(r5,rE);
            r6 = ADD_VEC_SS(r6,rD);
            r7 = MUL_VEC_SS(r7,rC);
            r8 = ADD_VEC_SS(r8,rF);
            r9 = MUL_VEC_SS(r9,rE);
            rA = ADD_VEC_SS(rA,rD);
            rB = MUL_VEC_SS(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SS(r0,r1);
    r2 = ADD_VEC_SS(r2,r3);
    r4 = ADD_VEC_SS(r4,r5);
    r6 = ADD_VEC_SS(r6,r7);
    r8 = ADD_VEC_SS(r8,r9);
    rA = ADD_VEC_SS(rA,rB);

    r0 = ADD_VEC_SS(r0,r2);
    r4 = ADD_VEC_SS(r4,r6);
    r8 = ADD_VEC_SS(r8,rA);

    r0 = ADD_VEC_SS(r0,r4);
    r0 = ADD_VEC_SS(r0,r8);

    float out = 0;
    SP_SCALAR_TYPE temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
double test_dp_scalar_VEC_24( uint64 iterations ){
    register DP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SD(0.01);
    r1 = SET_VEC_SD(0.02);
    r2 = SET_VEC_SD(0.03);
    r3 = SET_VEC_SD(0.04);
    r4 = SET_VEC_SD(0.05);
    r5 = SET_VEC_SD(0.06);
    r6 = SET_VEC_SD(0.07);
    r7 = SET_VEC_SD(0.08);
    r8 = SET_VEC_SD(0.09);
    r9 = SET_VEC_SD(0.10);
    rA = SET_VEC_SD(0.11);
    rB = SET_VEC_SD(0.12);
    rC = SET_VEC_SD(0.13);
    rD = SET_VEC_SD(0.14);
    rE = SET_VEC_SD(0.15);
    rF = SET_VEC_SD(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SD(r0,r1);
    r2 = ADD_VEC_SD(r2,r3);
    r4 = ADD_VEC_SD(r4,r5);
    r6 = ADD_VEC_SD(r6,r7);
    r8 = ADD_VEC_SD(r8,r9);
    rA = ADD_VEC_SD(rA,rB);

    r0 = ADD_VEC_SD(r0,r2);
    r4 = ADD_VEC_SD(r4,r6);
    r8 = ADD_VEC_SD(r8,rA);

    r0 = ADD_VEC_SD(r0,r4);
    r0 = ADD_VEC_SD(r0,r8);

    double out = 0;
    DP_SCALAR_TYPE temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
double test_dp_scalar_VEC_48( uint64 iterations ){
    register DP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SD(0.01);
    r1 = SET_VEC_SD(0.02);
    r2 = SET_VEC_SD(0.03);
    r3 = SET_VEC_SD(0.04);
    r4 = SET_VEC_SD(0.05);
    r5 = SET_VEC_SD(0.06);
    r6 = SET_VEC_SD(0.07);
    r7 = SET_VEC_SD(0.08);
    r8 = SET_VEC_SD(0.09);
    r9 = SET_VEC_SD(0.10);
    rA = SET_VEC_SD(0.11);
    rB = SET_VEC_SD(0.12);
    rC = SET_VEC_SD(0.13);
    rD = SET_VEC_SD(0.14);
    rE = SET_VEC_SD(0.15);
    rF = SET_VEC_SD(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SD(r0,r1);
    r2 = ADD_VEC_SD(r2,r3);
    r4 = ADD_VEC_SD(r4,r5);
    r6 = ADD_VEC_SD(r6,r7);
    r8 = ADD_VEC_SD(r8,r9);
    rA = ADD_VEC_SD(rA,rB);

    r0 = ADD_VEC_SD(r0,r2);
    r4 = ADD_VEC_SD(r4,r6);
    r8 = ADD_VEC_SD(r8,rA);

    r0 = ADD_VEC_SD(r0,r4);
    r0 = ADD_VEC_SD(r0,r8);

    double out = 0;
    DP_SCALAR_TYPE temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
double test_dp_scalar_VEC_96( uint64 iterations ){
    register DP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SD(0.01);
    r1 = SET_VEC_SD(0.02);
    r2 = SET_VEC_SD(0.03);
    r3 = SET_VEC_SD(0.04);
    r4 = SET_VEC_SD(0.05);
    r5 = SET_VEC_SD(0.06);
    r6 = SET_VEC_SD(0.07);
    r7 = SET_VEC_SD(0.08);
    r8 = SET_VEC_SD(0.09);
    r9 = SET_VEC_SD(0.10);
    rA = SET_VEC_SD(0.11);
    rB = SET_VEC_SD(0.12);
    rC = SET_VEC_SD(0.13);
    rD = SET_VEC_SD(0.14);
    rE = SET_VEC_SD(0.15);
    rF = SET_VEC_SD(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            r0 = MUL_VEC_SD(r0,rC);
            r1 = ADD_VEC_SD(r1,rD);
            r2 = MUL_VEC_SD(r2,rE);
            r3 = ADD_VEC_SD(r3,rF);
            r4 = MUL_VEC_SD(r4,rC);
            r5 = ADD_VEC_SD(r5,rD);
            r6 = MUL_VEC_SD(r6,rE);
            r7 = ADD_VEC_SD(r7,rF);
            r8 = MUL_VEC_SD(r8,rC);
            r9 = ADD_VEC_SD(r9,rD);
            rA = MUL_VEC_SD(rA,rE);
            rB = ADD_VEC_SD(rB,rF);

            r0 = ADD_VEC_SD(r0,rF);
            r1 = MUL_VEC_SD(r1,rE);
            r2 = ADD_VEC_SD(r2,rD);
            r3 = MUL_VEC_SD(r3,rC);
            r4 = ADD_VEC_SD(r4,rF);
            r5 = MUL_VEC_SD(r5,rE);
            r6 = ADD_VEC_SD(r6,rD);
            r7 = MUL_VEC_SD(r7,rC);
            r8 = ADD_VEC_SD(r8,rF);
            r9 = MUL_VEC_SD(r9,rE);
            rA = ADD_VEC_SD(rA,rD);
            rB = MUL_VEC_SD(rB,rC);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SD(r0,r1);
    r2 = ADD_VEC_SD(r2,r3);
    r4 = ADD_VEC_SD(r4,r5);
    r6 = ADD_VEC_SD(r6,r7);
    r8 = ADD_VEC_SD(r8,r9);
    rA = ADD_VEC_SD(rA,rB);

    r0 = ADD_VEC_SD(r0,r2);
    r4 = ADD_VEC_SD(r4,r6);
    r8 = ADD_VEC_SD(r8,rA);

    r0 = ADD_VEC_SD(r0,r4);
    r0 = ADD_VEC_SD(r0,r8);

    double out = 0;
    DP_SCALAR_TYPE temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

#if defined(ARM)
half test_hp_scalar_VEC_FMA_12( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SH(0.01);
    r1 = SET_VEC_SH(0.02);
    r2 = SET_VEC_SH(0.03);
    r3 = SET_VEC_SH(0.04);
    r4 = SET_VEC_SH(0.05);
    r5 = SET_VEC_SH(0.06);
    r6 = SET_VEC_SH(0.07);
    r7 = SET_VEC_SH(0.08);
    r8 = SET_VEC_SH(0.09);
    r9 = SET_VEC_SH(0.10);
    rA = SET_VEC_SH(0.11);
    rB = SET_VEC_SH(0.12);
    rC = SET_VEC_SH(0.13);
    rD = SET_VEC_SH(0.14);
    rE = SET_VEC_SH(0.15);
    rF = SET_VEC_SH(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SH(r0,r1);
    r2 = ADD_VEC_SH(r2,r3);
    r4 = ADD_VEC_SH(r4,r5);

    r0 = ADD_VEC_SH(r0,r6);
    r2 = ADD_VEC_SH(r2,r4);

    r0 = ADD_VEC_SH(r0,r2);

    half out = 0;
    half temp = r0;
    out = ADD_VEC_SH(out,temp);

    return out;
}

half test_hp_scalar_VEC_FMA_24( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SH(0.01);
    r1 = SET_VEC_SH(0.02);
    r2 = SET_VEC_SH(0.03);
    r3 = SET_VEC_SH(0.04);
    r4 = SET_VEC_SH(0.05);
    r5 = SET_VEC_SH(0.06);
    r6 = SET_VEC_SH(0.07);
    r7 = SET_VEC_SH(0.08);
    r8 = SET_VEC_SH(0.09);
    r9 = SET_VEC_SH(0.10);
    rA = SET_VEC_SH(0.11);
    rB = SET_VEC_SH(0.12);
    rC = SET_VEC_SH(0.13);
    rD = SET_VEC_SH(0.14);
    rE = SET_VEC_SH(0.15);
    rF = SET_VEC_SH(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SH(r0,r1);
    r2 = ADD_VEC_SH(r2,r3);
    r4 = ADD_VEC_SH(r4,r5);

    r0 = ADD_VEC_SH(r0,r6);
    r2 = ADD_VEC_SH(r2,r4);

    r0 = ADD_VEC_SH(r0,r2);

    half out = 0;
    half temp = r0;
    out = ADD_VEC_SH(out,temp);

    return out;
}

half test_hp_scalar_VEC_FMA_48( uint64 iterations ){
    register half r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SH(0.01);
    r1 = SET_VEC_SH(0.02);
    r2 = SET_VEC_SH(0.03);
    r3 = SET_VEC_SH(0.04);
    r4 = SET_VEC_SH(0.05);
    r5 = SET_VEC_SH(0.06);
    r6 = SET_VEC_SH(0.07);
    r7 = SET_VEC_SH(0.08);
    r8 = SET_VEC_SH(0.09);
    r9 = SET_VEC_SH(0.10);
    rA = SET_VEC_SH(0.11);
    rB = SET_VEC_SH(0.12);
    rC = SET_VEC_SH(0.13);
    rD = SET_VEC_SH(0.14);
    rE = SET_VEC_SH(0.15);
    rF = SET_VEC_SH(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            FMA_VEC_SH(r0,r0,r7,r9);
            FMA_VEC_SH(r1,r1,r8,rA);
            FMA_VEC_SH(r2,r2,r9,rB);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,rB,rD);
            FMA_VEC_SH(r5,r5,rC,rE);

            FMA_VEC_SH(r0,r0,rD,rF);
            FMA_VEC_SH(r1,r1,rC,rE);
            FMA_VEC_SH(r2,r2,rB,rD);
            FMA_VEC_SH(r3,r3,rA,rC);
            FMA_VEC_SH(r4,r4,r9,rB);
            FMA_VEC_SH(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SH(r0,r1);
    r2 = ADD_VEC_SH(r2,r3);
    r4 = ADD_VEC_SH(r4,r5);

    r0 = ADD_VEC_SH(r0,r6);
    r2 = ADD_VEC_SH(r2,r4);

    r0 = ADD_VEC_SH(r0,r2);

    half out = 0;
    half temp = r0;
    out = ADD_VEC_SH(out,temp);

    return out;
}

#else
float test_hp_scalar_VEC_FMA_12( uint64 iterations ){

    (void)iterations;
    return 0.0;
}

float test_hp_scalar_VEC_FMA_24( uint64 iterations ){

    (void)iterations;
    return 0.0;
}

float test_hp_scalar_VEC_FMA_48( uint64 iterations ){

    (void)iterations;
    return 0.0;
}
#endif

/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
float test_sp_scalar_VEC_FMA_12( uint64 iterations ){
    register SP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SS(0.01);
    r1 = SET_VEC_SS(0.02);
    r2 = SET_VEC_SS(0.03);
    r3 = SET_VEC_SS(0.04);
    r4 = SET_VEC_SS(0.05);
    r5 = SET_VEC_SS(0.06);
    r6 = SET_VEC_SS(0.07);
    r7 = SET_VEC_SS(0.08);
    r8 = SET_VEC_SS(0.09);
    r9 = SET_VEC_SS(0.10);
    rA = SET_VEC_SS(0.11);
    rB = SET_VEC_SS(0.12);
    rC = SET_VEC_SS(0.13);
    rD = SET_VEC_SS(0.14);
    rE = SET_VEC_SS(0.15);
    rF = SET_VEC_SS(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SS(r0,r1);
    r2 = ADD_VEC_SS(r2,r3);
    r4 = ADD_VEC_SS(r4,r5);

    r0 = ADD_VEC_SS(r0,r6);
    r2 = ADD_VEC_SS(r2,r4);

    r0 = ADD_VEC_SS(r0,r2);

    float out = 0;
    SP_SCALAR_TYPE temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
float test_sp_scalar_VEC_FMA_24( uint64 iterations ){
    register SP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SS(0.01);
    r1 = SET_VEC_SS(0.02);
    r2 = SET_VEC_SS(0.03);
    r3 = SET_VEC_SS(0.04);
    r4 = SET_VEC_SS(0.05);
    r5 = SET_VEC_SS(0.06);
    r6 = SET_VEC_SS(0.07);
    r7 = SET_VEC_SS(0.08);
    r8 = SET_VEC_SS(0.09);
    r9 = SET_VEC_SS(0.10);
    rA = SET_VEC_SS(0.11);
    rB = SET_VEC_SS(0.12);
    rC = SET_VEC_SS(0.13);
    rD = SET_VEC_SS(0.14);
    rE = SET_VEC_SS(0.15);
    rF = SET_VEC_SS(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SS(r0,r1);
    r2 = ADD_VEC_SS(r2,r3);
    r4 = ADD_VEC_SS(r4,r5);

    r0 = ADD_VEC_SS(r0,r6);
    r2 = ADD_VEC_SS(r2,r4);

    r0 = ADD_VEC_SS(r0,r2);

    float out = 0;
    SP_SCALAR_TYPE temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
float test_sp_scalar_VEC_FMA_48( uint64 iterations ){
    register SP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SS(0.01);
    r1 = SET_VEC_SS(0.02);
    r2 = SET_VEC_SS(0.03);
    r3 = SET_VEC_SS(0.04);
    r4 = SET_VEC_SS(0.05);
    r5 = SET_VEC_SS(0.06);
    r6 = SET_VEC_SS(0.07);
    r7 = SET_VEC_SS(0.08);
    r8 = SET_VEC_SS(0.09);
    r9 = SET_VEC_SS(0.10);
    rA = SET_VEC_SS(0.11);
    rB = SET_VEC_SS(0.12);
    rC = SET_VEC_SS(0.13);
    rD = SET_VEC_SS(0.14);
    rE = SET_VEC_SS(0.15);
    rF = SET_VEC_SS(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            FMA_VEC_SS(r0,r0,r7,r9);
            FMA_VEC_SS(r1,r1,r8,rA);
            FMA_VEC_SS(r2,r2,r9,rB);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,rB,rD);
            FMA_VEC_SS(r5,r5,rC,rE);

            FMA_VEC_SS(r0,r0,rD,rF);
            FMA_VEC_SS(r1,r1,rC,rE);
            FMA_VEC_SS(r2,r2,rB,rD);
            FMA_VEC_SS(r3,r3,rA,rC);
            FMA_VEC_SS(r4,r4,r9,rB);
            FMA_VEC_SS(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SS(r0,r1);
    r2 = ADD_VEC_SS(r2,r3);
    r4 = ADD_VEC_SS(r4,r5);

    r0 = ADD_VEC_SS(r0,r6);
    r2 = ADD_VEC_SS(r2,r4);

    r0 = ADD_VEC_SS(r0,r2);

    float out = 0;
    SP_SCALAR_TYPE temp = r0;
    out += ((float*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
double test_dp_scalar_VEC_FMA_12( uint64 iterations ){
    register DP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SD(0.01);
    r1 = SET_VEC_SD(0.02);
    r2 = SET_VEC_SD(0.03);
    r3 = SET_VEC_SD(0.04);
    r4 = SET_VEC_SD(0.05);
    r5 = SET_VEC_SD(0.06);
    r6 = SET_VEC_SD(0.07);
    r7 = SET_VEC_SD(0.08);
    r8 = SET_VEC_SD(0.09);
    r9 = SET_VEC_SD(0.10);
    rA = SET_VEC_SD(0.11);
    rB = SET_VEC_SD(0.12);
    rC = SET_VEC_SD(0.13);
    rD = SET_VEC_SD(0.14);
    rE = SET_VEC_SD(0.15);
    rF = SET_VEC_SD(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SD(r0,r1);
    r2 = ADD_VEC_SD(r2,r3);
    r4 = ADD_VEC_SD(r4,r5);

    r0 = ADD_VEC_SD(r0,r6);
    r2 = ADD_VEC_SD(r2,r4);

    r0 = ADD_VEC_SD(r0,r2);

    double out = 0;
    DP_SCALAR_TYPE temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
double test_dp_scalar_VEC_FMA_24( uint64 iterations ){
    register DP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SD(0.01);
    r1 = SET_VEC_SD(0.02);
    r2 = SET_VEC_SD(0.03);
    r3 = SET_VEC_SD(0.04);
    r4 = SET_VEC_SD(0.05);
    r5 = SET_VEC_SD(0.06);
    r6 = SET_VEC_SD(0.07);
    r7 = SET_VEC_SD(0.08);
    r8 = SET_VEC_SD(0.09);
    r9 = SET_VEC_SD(0.10);
    rA = SET_VEC_SD(0.11);
    rB = SET_VEC_SD(0.12);
    rC = SET_VEC_SD(0.13);
    rD = SET_VEC_SD(0.14);
    rE = SET_VEC_SD(0.15);
    rF = SET_VEC_SD(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SD(r0,r1);
    r2 = ADD_VEC_SD(r2,r3);
    r4 = ADD_VEC_SD(r4,r5);

    r0 = ADD_VEC_SD(r0,r6);
    r2 = ADD_VEC_SD(r2,r4);

    r0 = ADD_VEC_SD(r0,r2);

    double out = 0;
    DP_SCALAR_TYPE temp = r0;
    out += ((double*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
double test_dp_scalar_VEC_FMA_48( uint64 iterations ){
    register DP_SCALAR_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_SD(0.01);
    r1 = SET_VEC_SD(0.02);
    r2 = SET_VEC_SD(0.03);
    r3 = SET_VEC_SD(0.04);
    r4 = SET_VEC_SD(0.05);
    r5 = SET_VEC_SD(0.06);
    r6 = SET_VEC_SD(0.07);
    r7 = SET_VEC_SD(0.08);
    r8 = SET_VEC_SD(0.09);
    r9 = SET_VEC_SD(0.10);
    rA = SET_VEC_SD(0.11);
    rB = SET_VEC_SD(0.12);
    rC = SET_VEC_SD(0.13);
    rD = SET_VEC_SD(0.14);
    rE = SET_VEC_SD(0.15);
    rF = SET_VEC_SD(0.16);

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){

            /* The performance critical part */
            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            FMA_VEC_SD(r0,r0,r7,r9);
            FMA_VEC_SD(r1,r1,r8,rA);
            FMA_VEC_SD(r2,r2,r9,rB);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,rB,rD);
            FMA_VEC_SD(r5,r5,rC,rE);
            
            FMA_VEC_SD(r0,r0,rD,rF);
            FMA_VEC_SD(r1,r1,rC,rE);
            FMA_VEC_SD(r2,r2,rB,rD);
            FMA_VEC_SD(r3,r3,rA,rC);
            FMA_VEC_SD(r4,r4,r9,rB);
            FMA_VEC_SD(r5,r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_SD(r0,r1);
    r2 = ADD_VEC_SD(r2,r3);
    r4 = ADD_VEC_SD(r4,r5);

    r0 = ADD_VEC_SD(r0,r6);
    r2 = ADD_VEC_SD(r2,r4);

    r0 = ADD_VEC_SD(r0,r2);

    double out = 0;
    DP_SCALAR_TYPE temp = r0;
    out += ((double*)&temp)[0];

    return out;
}
