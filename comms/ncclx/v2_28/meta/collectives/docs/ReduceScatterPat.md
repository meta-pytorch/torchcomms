# PAT ReduceScatter Algorithm

## Overview

PAT (Pairwise Algorithm Tree) ReduceScatter implements a recursive halving algorithm that efficiently reduces and scatters data across ranks. For N ranks, each portion completes in log2(N) steps.

## Key Data Structures

### ncclPatStep (collectives.h)

```cpp
struct ncclPatStep {
  int recvDim, sendDim;      // Dimension for recv/send (-1 = none/local)
  int recvOffset, sendOffset; // Offset within peer buffer
  int stepOffset;            // Step offset for pipelining
  int postRecv, postSend;    // Post flags for completion
  int nelem, last, flags;    // Element count, completion, status
  bool isFinalWrite;         // True if final write for a chunk (apply division for AVG)
  size_t inpIx, outIx;       // Input/output buffer indices
};
```

### PatRSAlgorithm Class Members

```cpp
offset, end    // Current chunk range being processed
count          // Total elements per rank in input buffer
chunkCount     // Elements per chunk iteration
nelem          // Actual elements this iteration
rank, nranks   // This rank's ID and total rank count
nrPow2         // Next power of 2 >= nranks
aggFactor      // Aggregation factor (batches multiple steps)
aggDelta       // Step delta = nrPow2 / aggFactor
as             // Current aggregated step index
a              // Sub-step within aggregated step
phase          // Current algorithm phase (0-4)
scale          // Scaling factor for phases 2-3
```

## recvDim and sendDim Encoding

```
recvDim = -1  ->  No receive (use local data only as source)
recvDim >= 0  ->  Receive from peer along hypercube dimension N

sendDim = -1  ->  Write to LOCAL output buffer (userOutput + outIx)
sendDim >= 0  ->  Send to peer along hypercube dimension N
```

The dimension corresponds to hypercube edges. For 8 ranks:
- Dim 0: pairs (0,1), (2,3), (4,5), (6,7) - rank XOR 1
- Dim 1: pairs (0,2), (1,3), (4,6), (5,7) - rank XOR 2
- Dim 2: pairs (0,4), (1,5), (2,6), (3,7) - rank XOR 4

## The 5 Phases

The algorithm uses 5 phases organized into two groups:

### Primary Reduction (Phases 0-1)
Handles the main recursive halving, processing odd-indexed `as` values.

| Phase | Description | recvDim | sendDim |
|-------|-------------|---------|---------|
| **0** | Initial scatter: D2D copy from input, send to dim 0 peer | -1 (none) | 0 |
| **1** | Recursive halving: receive from dimension, reduce, forward to next dimension | `firstBitSet(s)` | `firstBitSet(s')` or -1 |

### Secondary Reduction (Phases 2-3)
Activated when `aggFactor > 1`. Forms a butterfly pattern at increasing scales to complete the reduction.

| Phase | Description | recvDim | sendDim |
|-------|-------------|---------|---------|
| **2** | Receive from dim 0 peer, reduce, forward to higher dimension | 0 | `firstBitSet(s)` or -1 |
| **3** | Receive from higher dimension, reduce, forward or write locally | `firstBitSet(s)` | `firstBitSet(s')` or -1 |

Phases 2-3 loop with `scale` doubling each iteration until `scale >= aggFactor`.

### Finalization (Phase 4)

| Phase | Description | recvDim | sendDim |
|-------|-------------|---------|---------|
| **4** | Final receive from dim 0, reduce, write to output buffer | 0 | -1 (local) |

## 8-Rank ReduceScatter Example

### Setup

```
Ranks: 0, 1, 2, 3, 4, 5, 6, 7
nrPow2 = 8
Dimensions: 0, 1, 2 (log2(8) = 3 dimensions)

Input: Each rank has input[0..7] (8 portions)
Output: Rank r gets reduced sum of all ranks' input[r]
```

### 3-Step Recursive Halving for portion[0] -> R0

```
STEP 1: Dim 0 exchange (pairs: 0<->1, 2<->3, 4<->5, 6<->7)
================================================================================

    R0          R1          R2          R3          R4          R5          R6          R7
   [0_0]       [0_1]       [0_2]       [0_3]       [0_4]       [0_5]       [0_6]       [0_7]
     |           |           |           |           |           |           |           |
     +-----+-----+           +-----+-----+           +-----+-----+           +-----+-----+
           |                       |                       |                       |
           v                       v                       v                       v
       [S0_{0,1}]             [S0_{2,3}]             [S0_{4,5}]             [S0_{6,7}]
        at R0                  at R2                  at R4                  at R6


STEP 2: Dim 1 exchange (pairs: 0<->2, 1<->3, 4<->6, 5<->7)
================================================================================

        R0                      R2                      R4                      R6
    [S0_{0,1}]             [S0_{2,3}]             [S0_{4,5}]             [S0_{6,7}]
         |                       |                       |                       |
         +-----------+-----------+                       +-----------+-----------+
                     |                                               |
                     v                                               v
              [S0_{0,1,2,3}]                                  [S0_{4,5,6,7}]
                 at R0                                           at R4


STEP 3: Dim 2 exchange (pairs: 0<->4, 1<->5, 2<->6, 3<->7)
================================================================================

                R0                                              R4
           [S0_{0,1,2,3}]                                 [S0_{4,5,6,7}]
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
                                  [S0_{all 8 ranks}]
                                      at R0
                                         |
                                    /8 (AVG)
                                         |
                                         v
                                   R0.OUTPUT = AVG
```

### All 8 Portions in Parallel (same 3 steps)

```
STEP 1 (Dim 0): Each dim0 pair reduces
--------------------------------------------------------------------------------
    portion[0]: R0,R1 -> R0          portion[1]: R0,R1 -> R1
    portion[2]: R2,R3 -> R2          portion[3]: R2,R3 -> R3
    portion[4]: R4,R5 -> R4          portion[5]: R4,R5 -> R5
    portion[6]: R6,R7 -> R6          portion[7]: R6,R7 -> R7

STEP 2 (Dim 1): Each dim1 pair reduces
--------------------------------------------------------------------------------
    portion[0]: R0,R2 -> R0          portion[1]: R1,R3 -> R1
    portion[2]: R0,R2 -> R2          portion[3]: R1,R3 -> R3
    portion[4]: R4,R6 -> R4          portion[5]: R5,R7 -> R5
    portion[6]: R4,R6 -> R6          portion[7]: R5,R7 -> R7

STEP 3 (Dim 2): Each dim2 pair reduces, FINAL destination reached
--------------------------------------------------------------------------------
    portion[0]: R0,R4 -> R0          portion[1]: R1,R5 -> R1
    portion[2]: R2,R6 -> R2          portion[3]: R3,R7 -> R3
    portion[4]: R0,R4 -> R4          portion[5]: R1,R5 -> R5
    portion[6]: R2,R6 -> R6          portion[7]: R3,R7 -> R7

    All portions complete in 3 steps! Apply /8 for AVG.
```


## Buffer Operations in Device Code (prims_simple.h)

```cpp
// Sources setup:
if (recv) {
    srcs[0] = peer->buff + recvOffset;     // Received data from peer, stored in tmp buffer
}
if (send && sendDim >= 0) {
    dsts[0] = peer->buff + sendOffset;     // Send tmp buffer
    srcs[1] = userInput + inpIx;           // Local contribution
}
if (sendDim < 0) {  // Local write (phase 4 or intermediate)
    dsts[0] = userOutput + outIx;          // Output buffer
    srcs[1] = userInput + inpIx;           // Local contribution
}

// Reduce: srcs[0] (received) + srcs[1] (local) -> dsts[0]
reduceCopy(..., nSrcs, srcs, 1, dsts, ...);
```
