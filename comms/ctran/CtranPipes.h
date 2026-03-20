// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/utils/commSpecs.h"

class CtranComm;
class CtranAlgo;

// Create and configure MultiPeerTransport on the CtranComm.
// exchange() is deferred to ctranInitPipesResources().
commResult_t ctranInitializePipes(CtranComm* comm);

// Wire SharedResource staging buffers as external data buffers to
// MultiPeerTransport and exchange handles. Must be called after both
// CtranAlgo (SharedResource) and MultiPeerTransport have been created.
commResult_t ctranInitPipesResources(CtranAlgo* algo);
