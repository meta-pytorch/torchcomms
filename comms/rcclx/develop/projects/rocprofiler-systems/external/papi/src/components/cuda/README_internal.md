# Cuda Component Native Events in PAPI

At current the Cuda component uses bits to create an Event identifier. The following internal README will be used to breakdown the encoding format.

# Event Identifier Encoding Format

## Unused bits

As of 02/02/25, there are a total of 2 unused bits. These bits can be used to create a new qualifier or can be used to extend the number of bits for an existing qualifier.

## STAT

3 bits are allocated for the statistic qualifier. ([0 - 7 stats]).

## Device

7 bits are allocated for the device which accounts for 128 total devices on a node (e.g. [0 - 127 devices]).

## Qlmask

2 bits are allocated for the qualifier mask.

## Nameid

18 bits are allocated for the nameid which will roughly account for greater than 260k Cuda native events per device on a node.

## Calculations for Bit Masks and Shifts

| #DEFINE      | Bits                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| EVENTS_WIDTH | `(sizeof(uint32_t) * 8)`                                                   |
| STAT_WIDTH   | `( 3)`                                                                     |
| DEVICE_WIDTH | `( 7)`                                                                     |
| QLMASK_WIDTH | `( 2)`                                                                     |
| NAMEID_WIDTH | `(18)`                                                                     |
| UNUSED_WIDTH | `(EVENTS_WIDTH - DEVICE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH - STAT_WIDTH)` |
| STAT_SHIFT   | `(EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH)`                               |
| DEVICE_SHIFT | `(EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH - DEVICE_WIDTH)`                |
| QLMASK_SHIFT | `(DEVICE_SHIFT - QLMASK_WIDTH)`                                            |
| NAMEID_SHIFT | `(QLMASK_SHIFT - NAMEID_WIDTH)`                                            |
| STAT_MASK    | `((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - STAT_WIDTH))   << STAT_SHIFT)`    |
| DEVICE_MASK  | `((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)`  |
| QLMASK_MASK  | `((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)`  |
| NAMEID_MASK  | `((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)`  |
| STAT_FLAG    | `STAT_FLAG  (0x2)`                                                         |
| DEVICE_FLAG  | `DEVICE_FLAG  (0x1)`                                                       |

**NOTE**: If adding a new qualifier, you must add it to the table found in the section titled [Calculations for Bit Masks and Shifts](#calculations-for-bit-masks-and-shifts) and account for this addition within `cupti_profiler.c`.
