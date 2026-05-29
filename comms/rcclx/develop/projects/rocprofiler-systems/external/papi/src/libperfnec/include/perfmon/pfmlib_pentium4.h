/*
 * Intel Pentium 4 PMU specific types and definitions (32 and 64 bit modes)
 *
 * Copyright (c) 2006 IBM Corp.
 * Contributed by Kevin Corry <kevcorry@us.ibm.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PFMLIB_PENTIUM4_H__
#define __PFMLIB_PENTIUM4_H__

#include <perfmon/pfmlib.h>

/* ESCR: Event Selection Control Register
 *
 * These registers are used to select which event to count along with options
 * for that event. There are (up to) 45 ESCRs, but each data counter is
 * restricted to a specific set of ESCRs.
 */

/**
 * pentium4_escr_value_t
 *
 * Bit-wise breakdown of the ESCR registers.
 *
 *    Bits     Description
 *   -------   -----------
 *   63 - 31   Reserved
 *   30 - 25   Event Select
 *   24 - 9    Event Mask
 *    8 - 5    Tag Value
 *      4      Tag Enable
 *      3      T0 OS - Enable counting in kernel mode (thread 0)
 *      2      T0 USR - Enable counting in user mode (thread 0)
 *      1      T1 OS - Enable counting in kernel mode (thread 1)
 *      0      T1 USR - Enable counting in user mode (thread 1)
 **/

#define EVENT_MASK_BITS 16
#define EVENT_SELECT_BITS 6

typedef union {
	unsigned long val;
	struct {
		unsigned long t1_usr:1;
		unsigned long t1_os:1;
		unsigned long t0_usr:1;
		unsigned long t0_os:1;
		unsigned long tag_enable:1;
		unsigned long tag_value:4;
		unsigned long event_mask:EVENT_MASK_BITS;
		unsigned long event_select:EVENT_SELECT_BITS;
		unsigned long reserved:1;
	} bits;
} pentium4_escr_value_t;

/* CCCR: Counter Configuration Control Register
 *
 * These registers are used to configure the data counters. There are 18
 * CCCRs, one for each data counter.
 */

/**
 * pentium4_cccr_value_t
 *
 * Bit-wise breakdown of the CCCR registers.
 *
 *    Bits     Description
 *   -------   -----------
 *   63 - 32   Reserved
 *     31      OVF - The data counter overflowed.
 *     30      Cascade - Enable cascading of data counter when alternate
 *             counter overflows.
 *   29 - 28   Reserved
 *     27      OVF_PMI_T1 - Generate interrupt for LP1 on counter overflow
 *     26      OVF_PMI_T0 - Generate interrupt for LP0 on counter overflow
 *     25      FORCE_OVF - Force interrupt on every counter increment
 *     24      Edge - Enable rising edge detection of the threshold comparison
 *             output for filtering event counts.
 *   23 - 20   Threshold Value - Select the threshold value for comparing to
 *             incoming event counts.
 *     19      Complement - Select how incoming event count is compared with
 *             the threshold value.
 *     18      Compare - Enable filtering of event counts.
 *   17 - 16   Active Thread - Only used with HT enabled.
 *             00 - None: Count when neither LP is active.
 *             01 - Single: Count when only one LP is active.
 *             10 - Both: Count when both LPs are active.
 *             11 - Any: Count when either LP is active.
 *   15 - 13   ESCR Select - Select which ESCR to use for selecting the
 *             event to count.
 *     12      Enable - Turns the data counter on or off.
 *   11 - 0    Reserved
 **/
typedef union {
	unsigned long val;
	struct {
		unsigned long reserved1:12;
		unsigned long enable:1;
		unsigned long escr_select:3;
		unsigned long active_thread:2;
		unsigned long compare:1;
		unsigned long complement:1;
		unsigned long threshold:4;
		unsigned long edge:1;
		unsigned long force_ovf:1;
		unsigned long ovf_pmi_t0:1;
		unsigned long ovf_pmi_t1:1;
		unsigned long reserved2:2;
		unsigned long cascade:1;
		unsigned long overflow:1;
	} bits;
} pentium4_cccr_value_t;

#endif /* __PFMLIB_PENTIUM4_H__ */
