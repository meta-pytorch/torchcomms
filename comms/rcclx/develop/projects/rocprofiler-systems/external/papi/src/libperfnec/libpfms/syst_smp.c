/*
 * syst_smp.c - system-wide monitoring for SMP machine using libpfms helper
 * library
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
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
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <inttypes.h>
#include <errno.h>
#include <stdarg.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#include <libpfms.h>


#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

static uint32_t
popcount(uint64_t c)
{
	uint32_t count = 0;

	for(; c; c>>=1) {
		if (c & 0x1)
			count++;
	}
	return count;
}

int
main(int argc, char **argv)
{
	pfarg_ctx_t ctx;
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_pmd_t *pd;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	uint64_t cpu_list;
	void *desc;
	unsigned int num_counters;
	uint32_t i, j, l, k, ncpus, npmds;
	size_t len;
	int ret;
	char *name;

	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("cannot initialize libpfm\n");

	if (pfms_initialize())
		fatal_error("cannot initialize libpfms\n");

	memset(&ctx, 0, sizeof(ctx));
	memset(pc, 0, sizeof(pc));

	ncpus = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
	if (ncpus == -1)
		fatal_error("cannot retrieve number of online processors\n");

	if (argc > 1) {
		cpu_list = strtoul(argv[1],NULL,0);
		if (popcount(cpu_list) > ncpus)
			fatal_error("too many processors specified\n");
	} else {
		cpu_list = ((1<<ncpus)-1);
	}

	/*
	 * use libpfm to prepare some decent PMC/PMD setup
	 */
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	pfm_get_num_counters(&num_counters);
	pfm_get_max_event_name_len(&len);

	name = malloc(len+1);
	if (name == NULL)
		fatal_error("cannot allocate memory for event name\n");


	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		fatal_error("cannot find inst retired event\n");

	i = 2;

	inp.pfp_dfl_plm = PFM_PLM3|PFM_PLM0;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}
	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * indicate we are using the monitors for a system-wide session.
	 * This may impact the way the library sets up the PMC values.
	 */
	inp.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));


	npmds = ncpus * inp.pfp_event_count;

	printf("ncpus=%u npmds=%u\n", ncpus, npmds);

	pd = calloc(npmds, sizeof(pfarg_pmd_t));
	if (pd == NULL)
		fatal_error("cannot allocate pd array\n");

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	/*
	 * We use inp.pfp_event_count PMD registers for our events per-CPU.
	 * We need to setup the PMDs we use. They are determined based on the
	 * PMC registers used. The following loop prepares the pd[] array
	 * for pfm_write_pmds(). With libpfms, on PMD write we need to pass
	 * only pfp_event_count PMD registers. But on PMD read, we need
	 * to pass pfp_event_count PMD registers per-CPU because libpfms
	 * does not aggregate counts. To prepapre for PMD read, we therefore
	 * propagate the PMD setup beyond just the first pfp_event_count
	 * elements of pd[].
	 */
	for(l=0, k= 0; l < ncpus; l++) {
		for (i=0; i < outp.pfp_pmd_count; i++, k++)
			pd[k].reg_num   = outp.pfp_pmds[i].reg_num;
	}

	/*
	 * create a context on all CPUs we asked for
	 *
	 * libpfms only works for system-wide, so we set the flag in
	 * the master context. the context argument is not modified by
	 * call.
	 *
	 * desc is an opaque descriptor used to identify session.
	 */
	ctx.ctx_flags = PFM_FL_SYSTEM_WIDE;

	ret = pfms_create(&cpu_list, 1, &ctx, NULL, &desc);
	if (ret == -1)
		fatal_error("create error %d\n", ret);

	/*
	 * program the PMC registers on all CPUs of interest
	 */
	ret = pfms_write_pmcs(desc, pc, outp.pfp_pmc_count);
	if (ret == -1)
		fatal_error("write_pmcs error %d\n", ret);

	/*
	 * program the PMD registers on all CPUs of interest
	 */
	ret = pfms_write_pmds(desc, pd, outp.pfp_pmd_count);
	if (ret == -1)
		fatal_error("write_pmds error %d\n", ret);

	/*
	 * load context on all CPUs of interest
	 */
	ret = pfms_load(desc);
	if (ret == -1)
		fatal_error("load error %d\n", ret);

	printf("monitoring for 10s on all CPUs\n");

	/*
	 * start monitoring on all CPUs of interest
	 */
	ret = pfms_start(desc);
	if (ret == -1)
		fatal_error("start error %d\n", ret);

	/*
	 * stop and listen to activity for 10s
	 */
	sleep(10);

	/*
	 * stop monitoring on all CPUs of interest
	 */
	ret = pfms_stop(desc);
	if (ret == -1)
		fatal_error("stop error %d\n", ret);

	/*
	 * read the PMD registers on all CPUs of interest.
	 * The pd[] array must be organized such that to
	 * read 2 PMDs on each CPU you need:
	 * 	- 2 * number of CPUs of interest
	 * 	- the first 2 elements of pd[] read on CPU0
	 * 	- the next  2 elements of pd[] read on CPU1
	 * 	- and so on
	 */
	ret = pfms_read_pmds(desc, pd, npmds);
	if (ret == -1)
		fatal_error("read_pmds error %d\n", ret);

	/*
	 * print per-CPU results
	 */
	for(j=0, k= 0; j < ncpus; j++) {
		for (i=0; i < inp.pfp_event_count; i++, k++) {
			pfm_get_full_event_name(&inp.pfp_events[i], name, len);
			printf("CPU%-3d PMD%u %20"PRIu64" %s\n",
					j,
					pd[k].reg_num,
					pd[k].reg_value,
					name);
		}
	}

	/*
	 * destroy context  on all CPUs of interest.
	 * After this call desc is invalid
	 */
	ret = pfms_close(desc);
	if (ret == -1)
		fatal_error("close error %d\n", ret);

	free(name);

	return 0;
}
