# Custom VARS

NCCLX CVARS - Strongly typed configurable knobs for NCCLX. All
configuration knobs are defined here, and can be used in source
file by including `nccl_cvars.h` and use typed CVAR by its name.

## User Guide

Refer to `nccl_cvars.yaml` for CVAR documentation and their default
values.

CVAR can be provided to program in two ways
1) Environment variable - e.g. `NCCL_DEBUG=warn nccl_allreduce_perf ...`
2) Config Variable - define in `/etc/nccl.conf` and it'll be picked up
   automatically by program.

Environment variable will take precedence over config variable. If not
specified in either, the default value will be used.

## Developer Guide

All CVARs are defined in `nccl_cvars.yaml`. To add a new CVAR:
1) Add the CVAR definition in `nccl_cvars.yaml`
2) Build any target that depends on `//comms/utils/cvars:ncclx-cvars` - the files will be auto-generated via genrule
3) Include `#include "comms/utils/cvars/nccl_cvars.h"` and use your CVAR in program

**Note:** `nccl_cvars.h` and `nccl_cvars.cc` are now generated at build time using a genrule.
They should **NOT** be manually edited or committed to the repository. The genrule automatically
generates these files from `nccl_cvars.yaml` using `extractcvars.py` whenever you build a target
that depends on the `ncclx-cvars` library.

To regenerate the files manually (for development/testing), you can run:
```bash
cd ~/fbsource/fbcode && buck2 run comms/utils/cvars:extractcvars
```

The CVAR is initialized as part of ncclInit and it is done by `initEnv` from `init.cc`. CVAR
must not be used before initialization.

## Changed NCCL CVAR Default values

NCCL_RAS_ENABLE - default value changed from 1 to 0
NCCL_CTRAN_IB_MAX_QPS - default value changed from 1 to 16
NCCL_CTRAN_IB_QP_MAX_MSGS - default value changed from 4 to 128
NCCL_CTRAN_IB_QP_SCALING_THRESHOLD - default value changed from 131072 to 524288
NCCL_CTRAN_IB_QP_CONFIG_XDC - default value changed from "" to "1048576,16,spray,128"
NCCL_CTRAN_IB_QP_CONFIG_XRACK - default value changed from "" to "1048576,16,spray,128"
NCCL_CTRAN_IB_QP_CONFIG_XZONE - default value changed from "" to "1048576,16,spray,128"
NCCL_CTRAN_IB_VC_MODE - default value changed from "spray" to "dqplb"

## NCCL Baseline Adapter

The NCCL Baseline Adapter API is designed to provide a similar interface to the baseline/third-party NCCL library's `ncclGetEnv` and `ncclLoadParam` functions.
