# `CtranIb` structure

This directory implements `CtranIb`, the InfiniBand backend used by ctran.
The class is composed of a few small, single-purpose helpers that all
plug into one main facade (`CtranIb`).

## At a glance

```
        ┌────────────────────────────────────────────────────────┐
        │                       CtranIb                          │
        │  (public IB ops: iput / iget / iflush / progress /     │
        │   isendCtrlMsg / irecvCtrlMsg / preConnect / regMem,   │
        │   lifecycle, epoch lock, CtranIbSingleton glue)        │
        │                                                        │
        │  members:                                              │
        │    ctran::ib::VcState                  vcState_        │
        │    std::unique_ptr<Bootstrap>          bootstrap_      │
        │    std::unique_ptr<BootstrapExternal>  externalBootstrap_ │
        └─────────┬──────────────┬──────────────────┬────────────┘
                  │              │                  │
                  ▼              ▼                  ▼
   ┌────────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
   │   ctran::ib::      │  │ ctran::ib::     │  │ ctran::ib::             │
   │      VcState       │  │   Bootstrap     │  │   BootstrapExternal     │
   │                    │  │                 │  │                         │
   │  - rank→VC map     │  │  - listen sock  │  │  - pendingVcs_ map      │
   │  - QP→VC map       │  │  - accept thrd  │  │  - mutex_               │
   │  - connectedPeer   │  │  - connect()    │  │  - getLocalVcId()       │
   │  - getVc/getVcByQp │  │  - shutdown()   │  │  - connectVc()          │
   │  - setupAndPublish │  │                 │  │  (allocated only in     │
   │                    │  │                 │  │   kExternal mode)       │
   └────────────────────┘  └────────┬────────┘  └─────────────┬───────────┘
            ▲                       │                         │
            │                       │ setupAndPublishVc       │ setupAndPublishVc
            └───────────────────────┴─────────────────────────┘
                              one-way arrow:
                       Bootstrap / BootstrapExternal
                              publish into VcState
```

## File map

| File | Purpose |
|---|---|
| `CtranIb.h` / `CtranIb.cc` | Facade. Owns `VcState`, `Bootstrap`, `BootstrapExternal`. Implements all public IB operations + lifecycle + epoch lock. Includes the `CtranIbSingleton` (per-process IBV resources). |
| `VcState.h` / `VcState.cc` | Per-peer VC registry. `rankToVcMap` / `qpToVcMap` (`folly::Synchronized<VcStateMaps>`), `connectedPeerMap` bitmap, `getVc`/`getVcByQp` templates, `setupAndPublishVc`. Pure data + lookup; no I/O. |
| `BootstrapInternal.h` / `BootstrapInternal.cc` | `ctran::ib::Bootstrap`. Owns the listen socket, accept thread, business-card exchange, and slow-path `connect()`. Publishes finished VCs into `VcState` via `setupAndPublishVc`. |
| `BootstrapExternal.h` / `BootstrapExternal.cc` | `ctran::ib::BootstrapExternal`. Owns the `pendingVcs_` map and the `getLocalVcId()` / `connectVc()` methods used by callsite-managed handshakes. Used only when `bootstrapMode == kExternal`; callers reach it via `CtranIb::externalBootstrap()`. |
| `CtranIbVc.{h,cc}` | `CtranIbVirtualConn` — one QP-group per peer. Owns the IB QPs (control / notify / atomic / data). Used by VC state map. |
| `CtranIbLocalVc.{h,cc}` | `LocalVirtualConn` — self-loop VC used by `iflush()` to force PCIe ordering. |
| `CtranIbBase.h` | Tiny POD types shared by everyone (`CtranIbDevice`, `CtranIbRequest`, `PendingOp`). |
| `CtranIbImpl.h` | Internal macros (`CTRAN_IB_PER_OBJ_LOCK_GUARD`, `CQE_ERROR_CHECK`) + `getRemoteKeysImpl`. |
| `CtranIbSingleton.h` | Process-global IBV PD/CQ context + async-event thread + memory-registration accounting. |
| `IbvWrap.{h,cc}` / `ibutils.{h,cc}` | Thin wrappers around libibverbs. |

## Namespace conventions

- `CtranIb`, `CtranIbVirtualConn`, `CtranIbBase`, `CtranIbRequest`,
  `QpUniqueId`, `CtranIbEpochRAII`, `CtranIbSingleton` — global namespace
  (legacy; the `CtranIb*` prefix is the namespace scope).
- `ctran::ib::VcState`, `ctran::ib::VcStateMaps`, `ctran::ib::Bootstrap`,
  `ctran::ib::BootstrapExternal`, `ctran::ib::LocalVirtualConn`,
  `ctran::ib::getRemoteKeysImpl` — new generic helpers live in
  `ctran::ib`.

## Lifecycle

```
CtranIb(...)                         ──►   init(...)
                                            │
                                            ▼
                              vcState_.init(owner, logData, nRanks)
                                            │
                              ┌─────────────┴──────────────┐
                              ▼                            ▼
              bootstrapMode == kExternal           bootstrapMode != kExternal
              externalBootstrap_ =                 bootstrap_ = make_unique<Bootstrap>(
                  make_unique<BootstrapExternal>(    vcState_, socketFactory_,
                      vcState_, devices, comm,       abortCtrl_, ncclLogData,
                      cudaDev, commHash, commDesc,   comm, devices,
                      ncclLogData,                   trafficClass, cudaDev,
                      trafficClass);                 rank, commHash, commDesc);
                                                   bootstrap_->start(qpServerAddr);
                                                     ↓
                                                   spawns accept thread
                                                     ↓
                                                   listenThread_ runs acceptLoop,
                                                   which calls
                                                   vcState_.setupAndPublishVc(...)
                                                   for each accepted peer

~CtranIb()                           ──►   if (bootstrap_) bootstrap_->shutdown();
                                            └─► closes listenSocket_, joins thread
                                          releaseRemoteTransStates(true)
                                            └─► vcState_.releaseAll(), free CQ, free localVc
                                          members destruct (in reverse declaration order):
                                            externalBootstrap_ → bootstrap_ → vcState_
```

The "slow path" for an outbound message (template in `CtranIb.h`):

```
isendCtrlMsgImpl / irecvCtrlMsgImpl
   │
   │  vc = vcState_.getVc(peerRank)
   │
   │  if (vc == nullptr  &&  rank < peerRank):
   │      bootstrap_->connect(peerRank, peerAddr)
   │                                                  │
   │                                                  │  open client socket,
   │                                                  │  exchange business cards,
   │                                                  │  vcState_.setupAndPublishVc(...)
   │
   │  vc = vcState_.getVc(peerRank)
   │  vc->isendCtrlMsg(...) / vc->irecvCtrlMsg(...)
```

The external-bootstrap path (driven by `RdmaTransport`):

```
RdmaTransport::bind()
   │
   │  return ib_->externalBootstrap()->getLocalVcId(peerRank)
   │      └─► constructs CtranIbVirtualConn, stashes it in pendingVcs_,
   │          returns the serialized local bus card
   │
RdmaTransport::connect(remoteBusCard)
   │
   │  ib_->externalBootstrap()->connectVc(remoteBusCard, peerRank)
   │      └─► pops pending VC, runs vc->setupVc(remoteBusCard),
   │          vcState_.setupAndPublishVc(...)
```

## Coupling rules

1. `VcState` depends on nothing in this directory except `CtranIbVirtualConn`
   (and `CommLogData` for logging). It does **not** include or
   forward-declare `Bootstrap` or `BootstrapExternal`.
2. `Bootstrap` depends on `VcState` (one method: `setupAndPublishVc`) and
   on the bootstrap I/O primitives in `comms/ctran/bootstrap/`. It does
   **not** depend on `CtranIb`.
3. `BootstrapExternal` depends on `VcState` (one method:
   `setupAndPublishVc`) and on `CtranIbVirtualConn`. It does **not**
   depend on `CtranIb`.
4. `CtranIb.h` includes `VcState.h` fully (`vcState_` is a value member)
   and `BootstrapInternal.h` so templates in the header can call
   `bootstrap_->connect(...)` directly. It forward-declares
   `BootstrapExternal` only (the external path is reached only via the
   `externalBootstrap()` accessor, which returns a forward-declared
   pointer).
5. `CtranIb.cc` includes `BootstrapInternal.h` and `BootstrapExternal.h`
   (so `~CtranIb()` and the `make_unique<BootstrapExternal>(...)` call
   in `init()` see complete types).

This means the entire external-bootstrap path — `BootstrapExternal`,
`CtranIb::externalBootstrap()`, and the `kExternal` enum value — lives
in three places (CtranIb.h field + accessor + `BootstrapExternal.{h,cc}`)
and can be removed in one shot once its only caller (`RdmaTransport`)
moves to Uniflow.
