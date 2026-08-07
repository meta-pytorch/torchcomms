# NCCL_CTRAN_IB_ENABLE_OOO_RQ — Design

Short design doc for the ctran-scoped out-of-order receive-queue opt-in.
For implementation details and file references, see `ooo_rq.md`.

## Purpose

Enable ctran data QPs to accept out-of-order packet arrival on the receiver
side, so `WRITE_WITH_IMM` traffic can be spread across multiple fabric paths
by adaptive-routing switches on CX8 NICs.

## Contract

```
NCCL_CTRAN_IB_ENABLE_OOO_RQ = false (default) → OOO_RQ off; ctran behaves exactly as before
NCCL_CTRAN_IB_ENABLE_OOO_RQ = true            → ctran negotiates OOO_RQ per connection;
                                                 fail-closed if any prereq is missing
```

Fail-closed means: if the user asks for OOO_RQ but the local NIC or the peer
can't support it, the connection is aborted with a WARN at BusCard receive
time. There is no silent fallback to non-OOO mode.

## Prerequisites checked per active device at init

1. **Provider is mlx5** (implicit — checked via `mlx5dv_query_device` succeeding)
2. **`ooo_recv_wrs_caps.max_rc >= 128`** — HCA firmware must expose enough
   per-QP OOO recv-WR capacity to cover ctran's `MAX_RECV_WR`

If any device fails either check, the local peer sets `localOooRq_ = false`
and advertises `oooRq = 0` in the BusCard. The remote peer will then observe
the mismatch on the receive side and abort with a WARN.

**`MLX5DV_QP_CREATE_OOO_DP` must be set on both sender and receiver.**
Per the man page: *"The flag [MLX5DV_QP_CREATE_OOO_DP], when set, must be set
both on the sender and receiver side of a QP."* Ctran naturally satisfies this
since both peers create their data QPs from the same code path with the same
cvar-driven decision.

## Design

### Where the mechanism lives

Ctran's IB backend already uses N data QPs per VC (`maxNumQps_`, default 16)
and sends each fragment as `WRITE_WITH_IMM` when in DQPLB mode. OOO_RQ integrates
by flipping a single flag on those data QPs at creation time — `MLX5DV_QP_CREATE_OOO_DP`.

Everything else about the send/receive pattern is unchanged:
- **Data QPs get OOO_DP.** N per VC.
- **Control / notify / atomic QPs do NOT get OOO_DP.** Their CQE handlers
  rely on FIFO order and would break under packet reordering.
- **DqplbSeqTracker still runs.** OOO_RQ handles within-QP packet reorder;
  cross-QP reassembly is still ctran's job via the seq numbers in IMM.

### Peer negotiation

Ctran's `BusCard` carries a single `oooRq` byte (0 or 1) advertising local
OOO_RQ eligibility. The field is always present on the wire — no
versioning.

Both peers exchange their advertisement. On the receive side, if
`NCCL_CTRAN_IB_ENABLE_OOO_RQ = true` locally and either side reports
`oooRq = 0`, the connection is aborted with a WARN — evaluated symmetrically
so both peers fail together and neither hangs waiting on a peer that already
gave up. Peers with the cvar off ignore the remote's advertisement (they
never asked for OOO_RQ).

### Data flow when OOO_RQ is enabled

```
Sender (ctran DQPLB):
  For each of 16 data QPs:
    post 1 WRITE_WITH_IMM(seq, notify_bit_on_last) per fragment
    ↓ (chunk = put_size / 16)
Fabric:
  Switch AR policy may spray packets across paths       ← depends on switch config,
                                                          out of ctran's scope
Receiver NIC (CX8 + OOO_RQ flag set on the RC QP):
  Buffers out-of-order packets                          ← what OOO_RQ enables
  Fires IMM CQE only after all packets of a WQE land
Receiver software (ctran):
  DqplbSeqTracker reassembles per-fragment order across the 16 QPs
  Fires application waitNotify() when notify_bit seq arrives in-order
```

The OOO_RQ change is purely a receiver-side NIC behavior toggle: absorb
per-QP in-fragment packet reordering. Nothing else in ctran changes.

## What the user sees

At init (per device):
```
CTRAN-IB: OOO_RQ detected on mlx5_2: max_rc=16384.
```

At each peer VC connection (when OOO_RQ negotiated successfully):
```
CTRAN-IB-VC: OOO_RQ ENABLED — creating 16 data QP(s) with MLX5DV_QP_CREATE_OOO_DP for peer 4
  (per-device oooRqSize checked >= MAX_RECV_WR=128). Control/notify/atomic QPs stay on the plain path.
```

When negotiation fails (fail-closed, at receive side on both peers):
```
CTRAN-IB-VC: NCCL_CTRAN_IB_ENABLE_OOO_RQ=true requested but negotiation failed
  with peer 4: localOooRq=0, remoteOooRq=0 (need both sides supported).
  Local per-device state: [dev=0 name=mlx5_2 oooRqSize=0] [dev=1 name=mlx5_3 oooRqSize=0].
  Aborting connection (no silent fallback).
```

## Failure modes and diagnostics

| Symptom | Likely cause | Action |
|---|---|---|
| `OOO_RQ detection FAILED ... symbol unavailable` | libmlx5 not loaded or dlvsym version-tag mismatch | Check `ldconfig -p \| grep mlx5`; confirm ibverbx loaded the symbol via unversioned dlsym |
| `did not populate OOO_RECV_WRS_CAPS mask bit` | Driver/firmware too old to expose caps | Check FW version on the CX8 NIC; upgrade if below 1.26-era |
| `oooRqSize=0` in fail-closed WARN | Hardware doesn't expose caps for that device | Confirm mlx5 provider and firmware version |
| Connection succeeds but no perf gain | Fabric switch not spraying WRITE_WITH_IMM on this TC | Escalate to network team for TC AR policy on the traffic class in use |

## Scope boundary

**In scope for this cvar:**
- QP-creation flag on ctran data QPs
- Bilateral peer negotiation via extended BusCard
- Fail-closed contract when requested-but-unsupported

**Out of scope (handled elsewhere or unchanged):**
- Stock NCCL net_ib OOO_RQ — separate cvar `NCCL_IB_OOO_RQ`
- Fabric switch AR policy — network-team owned
- CX8 NIC firmware behavior — NVIDIA owned
- Ctran spray mode — unaffected; OOO_RQ delivers no benefit to spray's
  plain-WRITE data path (separate notify QP has no fence pattern to eliminate)

## Interaction with other cvars

| Cvar | Interaction |
|---|---|
| `NCCL_CTRAN_IB_VC_MODE` | Recommended: `dqplb`. Spray mode gets no OOO_RQ benefit. |
| `NCCL_CTRAN_BACKENDS` | Must include `ib` for ctran to be the IB path |
| `NCCL_IB_OOO_RQ` | Independent — stock NCCL's opt-in, unaffected by this cvar |
| `NCCL_IB_ADAPTIVE_ROUTING` | Not read by ctran. Users still need fabric AR for the perf gain, but that's controlled by fabric config, not this cvar. |

## Verification

Four progressive signals, from weakest to strongest:

1. Init log shows `OOO_RQ detected on mlx5_X: max_rc=...` on every device
2. No `negotiation failed` WARN at connection time
3. Per-VC `OOO_RQ ENABLED — data QPs created with MLX5DV_QP_CREATE_OOO_DP` log
4. A/B perf comparison of `NCCL_CTRAN_IB_ENABLE_OOO_RQ=false` vs `=true` on
   identical workload shows measurable busbw delta at mid/large sizes

Signals 1-3 confirm the code path is exercised end-to-end (the QP flag reaches
the driver). Signal 4 confirms it delivers value on the specific fabric —
if it doesn't, the code is still correct but the fabric-team config is the
remaining lever.
