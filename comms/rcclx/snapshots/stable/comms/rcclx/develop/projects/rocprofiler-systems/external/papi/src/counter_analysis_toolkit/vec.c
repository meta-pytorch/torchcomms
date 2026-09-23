#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <papi.h>
#include "vec.h"
#include "cat_arch.h"
#include "vec_scalar_verify.h"

void vec_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".vec";
    char *papiFileName;

    (void)hw_desc;

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char)))) {
        return;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        goto error0;
    }

    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_add_named_event( EventSet, papi_event_name );
    if (retval != PAPI_OK ){
        goto error1;
    }

    // Header to label the columns in the output file.
    fprintf(ofp_papi, "# ExpectedInstrs EventCount\n");

#if defined(X86)

#if defined(AVX128_AVAIL)

    // HP Non-FMA instruction trials.
    fprintf(ofp_papi, "# HP Non-FMA Scalar\n");
    test_hp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# HP Non-FMA Vector AVX128\n");
    test_hp_x86_128B_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC( 96, ITER, EventSet, ofp_papi );

  #if defined(AVX256_AVAIL)
    fprintf(ofp_papi, "# HP Non-FMA Vector AVX256\n");
    test_hp_x86_256B_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC( 96, ITER, EventSet, ofp_papi );

    #if defined(AVX512_AVAIL)
    fprintf(ofp_papi, "# HP Non-FMA Vector AVX512\n");
    test_hp_x86_512B_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC( 96, ITER, EventSet, ofp_papi );
    #endif
  #endif

    // SP Non-FMA instruction trials.
    fprintf(ofp_papi, "# SP Non-FMA Scalar\n");
    test_sp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP Non-FMA Vector AVX128\n");
    test_sp_x86_128B_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC( 96, ITER, EventSet, ofp_papi );

  #if defined(AVX256_AVAIL)
    fprintf(ofp_papi, "# SP Non-FMA Vector AVX256\n");
    test_sp_x86_256B_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC( 96, ITER, EventSet, ofp_papi );

    #if defined(AVX512_AVAIL)
    fprintf(ofp_papi, "# SP Non-FMA Vector AVX512\n");
    test_sp_x86_512B_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC( 96, ITER, EventSet, ofp_papi );
    #endif
  #endif

    // DP Non-FMA instruction trials.
    fprintf(ofp_papi, "# DP Non-FMA Scalar\n");
    test_dp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP Non-FMA Vector AVX128\n");
    test_dp_x86_128B_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC( 96, ITER, EventSet, ofp_papi );

  #if defined(AVX256_AVAIL)
    fprintf(ofp_papi, "# DP Non-FMA Vector AVX256\n");
    test_dp_x86_256B_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC( 96, ITER, EventSet, ofp_papi );

    #if defined(AVX512_AVAIL)
    fprintf(ofp_papi, "# DP Non-FMA Vector AVX512\n");
    test_dp_x86_512B_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC( 96, ITER, EventSet, ofp_papi );
    #endif
  #endif

    // HP FMA instruction trials.
    fprintf(ofp_papi, "# HP FMA Scalar\n");
    test_hp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# HP FMA Vector AVX128\n");
    test_hp_x86_128B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

  #if defined(AVX256_AVAIL)
    fprintf(ofp_papi, "# HP FMA Vector AVX256\n");
    test_hp_x86_256B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    #if defined(AVX512_AVAIL)
    fprintf(ofp_papi, "# HP FMA Vector AVX512\n");
    test_hp_x86_512B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC_FMA( 48, ITER, EventSet, ofp_papi );
    #endif
  #endif

    // SP FMA instruction trials.
    fprintf(ofp_papi, "# SP FMA Scalar\n");
    test_sp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP FMA Vector AVX128\n");
    test_sp_x86_128B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

  #if defined(AVX256_AVAIL)
    fprintf(ofp_papi, "# SP FMA Vector AVX256\n");
    test_sp_x86_256B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    #if defined(AVX512_AVAIL)
    fprintf(ofp_papi, "# SP FMA Vector AVX512\n");
    test_sp_x86_512B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC_FMA( 48, ITER, EventSet, ofp_papi );
    #endif
  #endif

    // DP FMA instruction trials.
    fprintf(ofp_papi, "# DP FMA Scalar\n");
    test_dp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP FMA Vector AVX128\n");
    test_dp_x86_128B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

  #if defined(AVX256_AVAIL)
    fprintf(ofp_papi, "# DP FMA Vector AVX256\n");
    test_dp_x86_256B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    #if defined(AVX512_AVAIL)
    fprintf(ofp_papi, "# DP FMA Vector AVX512\n");
    test_dp_x86_512B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC_FMA( 48, ITER, EventSet, ofp_papi );
    #endif
  #endif

#else
    fprintf(stderr, "Vector FLOP benchmark is not supported on this architecture: AVX unavailable!\n");
#endif

#elif defined(ARM)

    // Non-FMA instruction trials.
    fprintf(ofp_papi, "# HP Non-FMA Scalar\n");
    test_hp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# HP Non-FMA Vector\n");
    test_hp_arm_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC( 96, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP Non-FMA Scalar\n");
    test_sp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP Non-FMA Vector\n");
    test_sp_arm_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC( 96, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP Non-FMA Scalar\n");
    test_dp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP Non-FMA Vector\n");
    test_dp_arm_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC( 96, ITER, EventSet, ofp_papi );

    // FMA instruction trials.
    fprintf(ofp_papi, "# HP FMA Scalar\n");
    test_hp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# HP FMA Vector\n");
    test_hp_arm_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP FMA Scalar\n");
    test_sp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP FMA Vector\n");
    test_sp_arm_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP FMA Scalar\n");
    test_dp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP FMA Vector\n");
    test_dp_arm_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#elif defined(POWER)

    // Non-FMA instruction trials.
    fprintf(ofp_papi, "# HP Non-FMA Scalar\n");
    test_hp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# HP Non-FMA Vector\n");
    test_hp_power_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_power_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_power_VEC( 96, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP Non-FMA Scalar\n");
    test_sp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP Non-FMA Vector\n");
    test_sp_power_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_power_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_power_VEC( 96, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP Non-FMA Scalar\n");
    test_dp_scalar_VEC_24( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_48( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_96( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP Non-FMA Vector\n");
    test_dp_power_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_power_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_power_VEC( 96, ITER, EventSet, ofp_papi );

    // FMA instruction trials.
    fprintf(ofp_papi, "# HP FMA Scalar\n");
    test_hp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_hp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# HP FMA Vector\n");
    test_hp_power_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_power_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_power_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP FMA Scalar\n");
    test_sp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_sp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# SP FMA Vector\n");
    test_sp_power_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_power_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_power_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP FMA Scalar\n");
    test_dp_scalar_VEC_FMA_12( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_FMA_24( ITER, EventSet, ofp_papi );
    test_dp_scalar_VEC_FMA_48( ITER, EventSet, ofp_papi );

    fprintf(ofp_papi, "# DP FMA Vector\n");
    test_dp_power_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_power_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_power_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#endif

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }
    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

error1:
    fclose(ofp_papi);
error0:
    free(papiFileName);
    return;
}
