// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/CtranComm.h"

class CtranPersistentRequest;

namespace ctran {
// Windowed-capture build/teardown for the ctgraph AllGather path, defined in
// AllGatherPWithWin.cc. createAllGatherPWithWindow returns a persistent request
// whose AlgoImpl owns the NVL window + split subcomm (nvlWin / nvlComm); the
// request must outlive every graph replay, so tear it down with
// destroyAllGatherPWithWindow at comm destroy. Internal to the windowed-capture
// implementation, not part of the public ctran API.
commResult_t createAllGatherPWithWindow(
    CtranComm* comm,
    void* recvbuff,
    size_t recvBytes,
    cudaStream_t stream,
    CtranPersistentRequest** out);
commResult_t destroyAllGatherPWithWindow(CtranPersistentRequest* request);
} // namespace ctran
