#pragma once

#include "rccl/rccl.h"

#if NCCL_VERSION_CODE < 22400
#define ncclFloat8e4m3 ncclFp8E4M3
#define ncclFloat8e5m2 ncclFp8E5M2
#endif
