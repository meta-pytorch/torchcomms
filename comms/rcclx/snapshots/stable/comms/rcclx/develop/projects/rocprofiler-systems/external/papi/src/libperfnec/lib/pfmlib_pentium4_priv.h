/*
 * Copyright (c) 2006 IBM Corp.
 * Contributed by Kevin Corry <kevcorry@us.ibm.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * pfmlib_pentium4_priv.h
 *
 * Structures and definitions for use in the Pentium4/Xeon/EM64T libpfm code.
 */

#ifndef _PFMLIB_PENTIUM4_PRIV_H_
#define _PFMLIB_PENTIUM4_PRIV_H_

/**
 * pentium4_escr_reg_t
 *
 * Describe one ESCR register.
 *
 * "pentium4_escrs" is a flat array of these structures
 * that defines all the ESCRs.
 *
 * @name: ESCR's name
 * @pmc: Perfmon's PMC number for this ESCR.
 * @allowed_cccrs: Array of CCCR numbers that can be used with this ESCR. A
 *                 positive value is an index into the pentium4_ccrs array.
 *                 A value of -1 indicates that slot is unused.
 **/

#define MAX_CCCRS_PER_ESCR 3

typedef struct {
	char *name;
	int pmc;
	int allowed_cccrs[MAX_CCCRS_PER_ESCR];
} pentium4_escr_reg_t;


/* CCCR: Counter Configuration Control Register
 *
 * These registers are used to configure the data counters. There are 18
 * CCCRs, one for each data counter.
 */

/**
 * pentium4_cccr_reg_t
 *
 * Describe one CCCR register.
 *
 * "pentium4_cccrs" is a flat array of these structures
 * that defines all the CCCRs.
 *
 * @name: CCCR's name
 * @pmc: Perfmon's PMC number for this CCCR
 * @pmd: Perfmon's PMD number for the associated data counter. Every CCCR has
 *       exactly one counter.
 * @allowed_escrs: Array of ESCR numbers that can be used with this CCCR. A
 *                 positive value is an index into the pentium4_escrs array.
 *                 A value of -1 indicates that slot is unused. The index into
 *                 this array is the value to use in the escr_select portion
 *                 of the CCCR value.
 **/

#define MAX_ESCRS_PER_CCCR 8

typedef struct {
	char *name;
	int pmc;
	int pmd;
	int allowed_escrs[MAX_ESCRS_PER_CCCR];
} pentium4_cccr_reg_t;

/**
 * pentium4_replay_regs_t
 *
 * Describe one pair of PEBS registers for use with the replay_event event.
 *
 * "p4_replay_regs" is a flat array of these structures
 * that defines all the PEBS pairs per Table A-10 of 
 * the Intel System Programming Guide Vol 3B.
 *
 * @enb:      value for the PEBS_ENABLE register for a given replay metric.
 * @mat_vert: value for the PEBS_MATRIX_VERT register for a given metric.
 *            The replay_event event defines a series of virtual mask bits
 *            that serve as indexes into this array. The values at that index
 *            provide information programmed into the PEBS registers to count
 *            specific metrics available to the replay_event event.
 **/

typedef struct {
	int enb;
	int mat_vert;
} pentium4_replay_regs_t;

/**
 * pentium4_pmc_t
 *
 * Provide a mapping from PMC number to the type of control register and
 * its index within the appropriate array.
 *
 * @name: Name
 * @type: PENTIUM4_PMC_TYPE_ESCR or PENTIUM4_PMC_TYPE_CCCR
 * @index: Index into the pentium4_escrs array or the pentium4_cccrs array.
 **/
typedef struct {
	char *name;
	int type;
	int index;
} pentium4_pmc_t;

#define PENTIUM4_PMC_TYPE_ESCR 1
#define PENTIUM4_PMC_TYPE_CCCR 2

/**
 * pentium4_event_mask_t
 *
 * Defines one bit of the event-mask for one Pentium4 event.
 *
 * @name: Event mask name
 * @desc: Event mask description
 * @bit: The bit position within the event_mask field.
 **/
typedef struct {
	char *name;
	char *desc;
	unsigned int bit;
} pentium4_event_mask_t;

/**
 * pentium4_event_t
 *
 * Describe one event that can be counted on Pentium4/EM64T.
 *
 * "pentium4_events" is a flat array of these structures that defines
 * all possible events.
 *
 * @name: Event name
 * @desc: Event description
 * @event_select: Value for the 'event_select' field in the ESCR (bits [31:25]).
 * @escr_select: Value for the 'escr_select' field in the CCCR (bits [15:13]).
 * @allowed_escrs: Numbers for ESCRs that can be used to count this event. A
 *                 positive value is an index into the pentium4_escrs array.
 *                 A value of -1 means that slot is not used.
 * @event_masks: Array of descriptions of available masks for this event.
 *               Array elements with a NULL 'name' field are unused.
 **/

#define MAX_ESCRS_PER_EVENT 2

typedef struct {
	char *name;
	char *desc;
	unsigned int event_select;
	unsigned int escr_select;
	int allowed_escrs[MAX_ESCRS_PER_EVENT];
	pentium4_event_mask_t event_masks[EVENT_MASK_BITS];
} pentium4_event_t;

#endif

