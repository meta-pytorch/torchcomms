#ifndef __FORCE_INIT_H__
#define __FORCE_INIT_H__

static inline void force_rocm_smi_init(int cid)
{
    int ntv_code = PAPI_NATIVE_MASK;
    PAPI_enum_cmp_event(&ntv_code, PAPI_ENUM_FIRST, cid);
}

#endif
