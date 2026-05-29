#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <inttypes.h>
#include <syscall.h>
#include <errno.h>
#include <stdarg.h>

#include <perfmon/perfmon.h>

#include "libpfms.h"

//#define dprint(format, arg...) fprintf(stderr, "%s.%d: " format , __FUNCTION__ , __LINE__, ## arg)
#define dprint(format, arg...)

typedef enum {	CMD_NONE,
		CMD_CTX,
		CMD_LOAD,
		CMD_UNLOAD,
		CMD_WPMCS,
		CMD_WPMDS,
		CMD_RPMDS,
		CMD_STOP,
		CMD_START,
		CMD_CLOSE
} pfms_cmd_t;

typedef struct _barrier {
	pthread_mutex_t mutex;
	pthread_cond_t	cond;
	uint32_t	counter;
	uint32_t	max;
	uint64_t	generation; /* avoid race condition on wake-up */
} barrier_t;

typedef struct {
	uint32_t	cpu;
	uint32_t	fd;
	void		*smpl_vaddr;
	size_t		smpl_buf_size;
} pfms_cpu_t;

typedef struct _pfms_thread {
	uint32_t	cpu;
	pfms_cmd_t	cmd;
	void		*data;
	uint32_t	ndata;
	sem_t		cmd_sem;
	int		ret;
	pthread_t	tid;
	barrier_t	*barrier; 
} pfms_thread_t;

typedef struct  {
	barrier_t	barrier;
	uint32_t	ncpus;
} pfms_session_t;

static uint32_t	ncpus;
static pfms_thread_t	*tds;
static pthread_mutex_t  tds_lock = PTHREAD_MUTEX_INITIALIZER;

static int
barrier_init(barrier_t *b, uint32_t count)
{
	int r;

	r = pthread_mutex_init(&b->mutex, NULL);
	if (r == -1) return -1;

	r = pthread_cond_init(&b->cond, NULL);
	if (r == -1) return -1;

	b->max = b->counter = count;
	b->generation = 0;

	return 0;
}

static void
cleanup_barrier(void *arg)
{
	barrier_t *b = (barrier_t *)arg;
	int r;
	r = pthread_mutex_unlock(&b->mutex);
	dprint("free barrier mutex r=%d\n", r);
	(void) r;
}

static int
barrier_wait(barrier_t *b)
{
	uint64_t generation;
	int oldstate;

	pthread_cleanup_push(cleanup_barrier, b);

	pthread_mutex_lock(&b->mutex);

	pthread_testcancel();

	if (--b->counter == 0) {

		/* reset barrier */
		b->counter = b->max;
		/*
		 * bump generation number, this avoids thread getting stuck in the
		 * wake up loop below in case a thread just out of the barrier goes
		 * back in right away before all the thread from the previous "round"
		 * have "escaped".
		 */
		b->generation++;

		pthread_cond_broadcast(&b->cond);
	} else {

		generation = b->generation;

		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &oldstate);

		while (b->counter != b->max && generation == b->generation) {
			pthread_cond_wait(&b->cond, &b->mutex);
		}

		pthread_setcancelstate(oldstate, NULL);
	}
	pthread_mutex_unlock(&b->mutex);

	pthread_cleanup_pop(0);

	return 0;
}

/*
 * placeholder for pthread_setaffinity_np(). This stuff is ugly
 * and I could not figure out a way to get it compiled while also preserving
 * the pthread_*cancel(). There are issues with LinuxThreads and NPTL. I
 * decided to quit on this and implement my own affinity call until this 
 * settles.
 */
static int
pin_cpu(uint32_t cpu)
{
	uint64_t *mask;
	size_t size;
	pid_t pid;
	int ret;

	pid = syscall(__NR_gettid);

	size = ncpus * sizeof(uint64_t);

	mask = calloc(1, size);
	if (mask == NULL) {
		dprint("CPU%u: cannot allocate bitvector\n", cpu);
		return -1;
	}
	mask[cpu>>6] = 1ULL << (cpu & 63);

	ret = syscall(__NR_sched_setaffinity, pid, size, mask);

	free(mask);

	return ret;
}

static void
pfms_thread_mainloop(void *arg)
{
	long k = (long )arg;
	uint32_t mycpu = (uint32_t)k;
	pfarg_ctx_t myctx, *ctx;
	pfarg_load_t load_args;
	int fd = -1;
	pfms_thread_t *td;
	sem_t *cmd_sem;
	int ret = 0;

	memset(&load_args, 0, sizeof(load_args));
	load_args.load_pid = mycpu;
	td = tds+mycpu;

	ret = pin_cpu(mycpu);
	dprint("CPU%u wthread created and pinned ret=%d\n", mycpu, ret);

	cmd_sem = &tds[mycpu].cmd_sem;

	for(;;) {
		dprint("CPU%u waiting for cmd\n", mycpu);

		sem_wait(cmd_sem);

		switch(td->cmd) {
			case CMD_NONE:
				ret = 0;
				break;

			case CMD_CTX:

				/*
				 * copy context to get private fd
				 */
				ctx = td->data;
				myctx = *ctx;

				fd = pfm_create_context(&myctx, NULL, NULL, 0);
				ret = fd < 0 ? -1 : 0;
				dprint("CPU%u CMD_CTX ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;

			case CMD_LOAD:
				ret = pfm_load_context(fd, &load_args);
				dprint("CPU%u CMD_LOAD ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_UNLOAD:
				ret = pfm_unload_context(fd);
				dprint("CPU%u CMD_UNLOAD ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_START:
				ret = pfm_start(fd, NULL);
				dprint("CPU%u CMD_START ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_STOP:
				ret = pfm_stop(fd);
				dprint("CPU%u CMD_STOP ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_WPMCS:
				ret = pfm_write_pmcs(fd,(pfarg_pmc_t *)td->data, td->ndata);
				dprint("CPU%u CMD_WPMCS ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_WPMDS:
				ret = pfm_write_pmds(fd,(pfarg_pmd_t *)td->data, td->ndata);
				dprint("CPU%u CMD_WPMDS ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_RPMDS:
				ret = pfm_read_pmds(fd,(pfarg_pmd_t *)td->data, td->ndata);
				dprint("CPU%u CMD_RPMDS ret=%d errno=%d fd=%d\n", mycpu, ret, errno, fd);
				break;
			case CMD_CLOSE:
				dprint("CPU%u CMD_CLOSE fd=%d\n", mycpu, fd);
				ret = close(fd);
				fd = -1;
				break;
			default:
				break;
		}
		td->ret = ret;

		dprint("CPU%u td->ret=%d\n", mycpu, ret);

		barrier_wait(td->barrier);
	}
}

static int
create_one_wthread(int cpu)
{
	int ret;

	sem_init(&tds[cpu].cmd_sem, 0, 0);

	ret = pthread_create(&tds[cpu].tid, 
			     NULL, 
			     (void *(*)(void *))pfms_thread_mainloop,
			     (void *)(long)cpu);
	return ret;
}

/*
 * must be called with tds_lock held
 */
static int
create_wthreads(uint64_t *cpu_list, uint32_t n)
{
	uint64_t v;
	uint32_t i,k, cpu;
	int ret = 0;

	for(k=0, cpu = 0; k < n; k++, cpu+= 64) {
		v = cpu_list[k];
		for(i=0; v && i < 63; i++, v>>=1, cpu++) {
			if ((v & 0x1) && tds[cpu].tid == 0) {
				ret = create_one_wthread(cpu);
				if (ret) break;
			}
		}
	}

	if (ret)
		dprint("cannot create wthread on CPU%u\n", cpu);

	return ret;
}

int
pfms_initialize(void)
{
	printf("cpu_t=%zu thread=%zu session_t=%zu\n",
		sizeof(pfms_cpu_t),
		sizeof(pfms_thread_t),
		sizeof(pfms_session_t));

	ncpus = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
	if (ncpus == -1) {
		dprint("cannot retrieve number of online processors\n");
		return -1;
	}

	dprint("configured for %u CPUs\n", ncpus);

	/*
	 * XXX: assuming CPU are contiguously indexed
	 */
	tds = calloc(ncpus, sizeof(*tds));
	if (tds == NULL) {
		dprint("cannot allocate thread descriptors\n");
		return -1;
	}
	return 0;
}

int
pfms_create(uint64_t *cpu_list, size_t n, pfarg_ctx_t *ctx, pfms_ovfl_t *ovfl, void **desc)
{
	uint64_t v;
	size_t k, i;
	uint32_t num, cpu;
	pfms_session_t *s;
	int ret;

	if (cpu_list == NULL || n == 0 || ctx == NULL || desc == NULL) {
		dprint("invalid parameters\n");
		return -1;
	}

	if ((ctx->ctx_flags & PFM_FL_SYSTEM_WIDE) == 0) {
		dprint("only works for system wide\n");
		return -1;
	}

	*desc = NULL;

	/*
	 * XXX: assuming CPU are contiguously indexed
	 */
	num = 0;
	for(k=0, cpu = 0; k < n; k++, cpu+=64) {
		v = cpu_list[k];
		for(i=0; v && i < 63; i++, v>>=1, cpu++) {
			if (v & 0x1) {
				if (cpu >= ncpus) {
					dprint("unavailable CPU%u\n", cpu);
					return -1;
				}
				num++;
			}
		}
	}

	if (num == 0)
		return 0;

	s = calloc(1, sizeof(*s));
	if (s == NULL) {
		dprint("cannot allocate %u contexts\n", num);
		return -1;
	}
	s->ncpus = num;

	printf("%u-way  session\n", num);

	/*
	 * +1 to account for main thread waiting
	 */
	ret = barrier_init(&s->barrier, num + 1);
	if (ret) {
		dprint("cannot init barrier\n");
		goto error_free;
	}

	/*
	 * lock thread descriptor table, no other create_session, close_session
	 * can occur
	 */
	pthread_mutex_lock(&tds_lock);

	if (create_wthreads(cpu_list, n))
		goto error_free_unlock;

	/*
	 * check all needed threads are available
	 */
	for(k=0, cpu = 0; k < n; k++, cpu += 64) {
		v = cpu_list[k];
		for(i=0; v && i < 63; i++, v>>=1, cpu++) {
			if (v & 0x1) {
				if (tds[cpu].barrier) {
					dprint("CPU%u already managing a session\n", cpu);
					goto error_free_unlock;
				}

			}
		}
	}

	/*
	 * send create context order
	 */
	for(k=0, cpu = 0; k < n; k++, cpu += 64) {
		v = cpu_list[k];
		for(i=0; v && i < 63; i++, v>>=1, cpu++) {
			if (v & 0x1) {
				tds[cpu].cmd  = CMD_CTX;
				tds[cpu].data = ctx;
				tds[cpu].barrier = &s->barrier;
				sem_post(&tds[cpu].cmd_sem);
			}
		}
	}
	barrier_wait(&s->barrier);

	ret = 0;

	/*
	 * check for errors
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			ret = tds[k].ret;
			if (ret)
				break;
		}
	}
	/*
	 * undo if error found
	 */
	if (k < ncpus) {
		for(k=0; k < ncpus; k++) {
			if (tds[k].barrier == &s->barrier) {
				if (tds[k].ret == 0) {
					tds[k].cmd = CMD_CLOSE;
					sem_post(&tds[k].cmd_sem);
				}
				/* mark as free */
				tds[k].barrier = NULL;
			}
		}
	}
	pthread_mutex_unlock(&tds_lock);

	if (ret == 0) *desc = s;

	return ret ? -1 : 0;

error_free_unlock:
	pthread_mutex_unlock(&tds_lock);

error_free:
	free(s);
	return -1;
}

int
pfms_load(void *desc)
{
	uint32_t k;
	pfms_session_t *s;
	int ret;

	if (desc == NULL) {
		dprint("invalid parameters\n");
		return -1;
	}
	s = (pfms_session_t *)desc;

	if (s->ncpus == 0) {
		dprint("invalid session content 0 CPUS\n");
		return -1;
	}
	/*
	 * send create context order
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			tds[k].cmd  = CMD_LOAD;
			sem_post(&tds[k].cmd_sem);
		}
	}

	barrier_wait(&s->barrier);

	ret = 0;

	/*
	 * check for errors
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			ret = tds[k].ret;
			if (ret) {
				dprint("failure on CPU%u\n", k);
				break;
			}
		}
	}

	/*
	 * if error, unload all others
	 */
	if (k < ncpus) {
		for(k=0; k < ncpus; k++) {
			if (tds[k].barrier == &s->barrier) {
				if (tds[k].ret == 0) {
					tds[k].cmd = CMD_UNLOAD;
					sem_post(&tds[k].cmd_sem);
				}
			}
		}
	}
	return ret ? -1 : 0;
}

static int
__pfms_do_simple_cmd(pfms_cmd_t cmd, void *desc, void *data, uint32_t n)
{
	size_t k;
	pfms_session_t *s;
	int ret;

	if (desc == NULL) {
		dprint("invalid parameters\n");
		return -1;
	}
	s = (pfms_session_t *)desc;

	if (s->ncpus == 0) {
		dprint("invalid session content 0 CPUS\n");
		return -1;
	}
	/*
	 * send create context order
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			tds[k].cmd  = cmd;
			tds[k].data = data;
			tds[k].ndata = n;
			sem_post(&tds[k].cmd_sem);
		}
	}
	barrier_wait(&s->barrier);

	ret = 0;

	/*
	 * check for errors
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			ret = tds[k].ret;
			if (ret) {
				dprint("failure on CPU%zu\n", k);
				break;
			}
		}
	}
	/*
	 * simple commands cannot be undone
	 */
	return ret ? -1 : 0;
}

int
pfms_unload(void *desc)
{
	return __pfms_do_simple_cmd(CMD_UNLOAD, desc, NULL, 0);
}

int
pfms_start(void *desc)
{
	return __pfms_do_simple_cmd(CMD_START, desc, NULL, 0);
}

int
pfms_stop(void *desc)
{
	return __pfms_do_simple_cmd(CMD_STOP, desc, NULL, 0);
}

int
pfms_write_pmcs(void *desc, pfarg_pmc_t *pmcs, uint32_t n)
{
	return __pfms_do_simple_cmd(CMD_WPMCS, desc, pmcs, n);
}

int
pfms_write_pmds(void *desc, pfarg_pmd_t *pmds, uint32_t n)
{
	return __pfms_do_simple_cmd(CMD_WPMDS, desc, pmds, n);
}

int
pfms_close(void *desc)
{
	size_t k;
	pfms_session_t *s;
	int ret;

	if (desc == NULL) {
		dprint("invalid parameters\n");
		return -1;
	}
	s = (pfms_session_t *)desc;

	if (s->ncpus == 0) {
		dprint("invalid session content 0 CPUS\n");
		return -1;
	}

	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			tds[k].cmd  = CMD_CLOSE;
			sem_post(&tds[k].cmd_sem);
		}
	}
	barrier_wait(&s->barrier);

	ret = 0;

	pthread_mutex_lock(&tds_lock);
	/*
	 * check for errors
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			if (tds[k].ret) {
				dprint("failure on CPU%zu\n", k);
			}
			ret |= tds[k].ret;
			tds[k].barrier = NULL;
		}
	}

	pthread_mutex_unlock(&tds_lock);

	free(s);

	/*
	 * XXX: we cannot undo close
	 */
	return ret ? -1 : 0;
}

int
pfms_read_pmds(void *desc, pfarg_pmd_t *pmds, uint32_t n)
{
	pfms_session_t *s;
	uint32_t k, pmds_per_cpu;
	int ret;

	if (desc == NULL) {
		dprint("invalid parameters\n");
		return -1;
	}
	s = (pfms_session_t *)desc;

	if (s->ncpus == 0) {
		dprint("invalid session content 0 CPUS\n");
		return -1;
	}
	if (n % s->ncpus) {
		dprint("invalid number of pfarg_pmd_t provided, must be multiple of %u\n", s->ncpus);
		return -1;
	}
	pmds_per_cpu = n / s->ncpus;

	dprint("n=%u ncpus=%u per_cpu=%u\n", n, s->ncpus, pmds_per_cpu);

	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			tds[k].cmd  = CMD_RPMDS;
			tds[k].data = pmds;
			tds[k].ndata= pmds_per_cpu;
			sem_post(&tds[k].cmd_sem);
			pmds += pmds_per_cpu;
		}
	}
	barrier_wait(&s->barrier);

	ret = 0;

	/*
	 * check for errors
	 */
	for(k=0; k < ncpus; k++) {
		if (tds[k].barrier == &s->barrier) {
			ret = tds[k].ret;
			if (ret) {
				dprint("failure on CPU%u\n", k);
				break;
			}
		}
	}
	/*
	 * cannot undo pfm_read_pmds
	 */
	return ret ? -1 : 0;
}
#if 0

/*
 * beginning of test program
 */
#include <perfmon/pfmlib.h>

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
	uint32_t i, j, k, l, ncpus, npmds;
	size_t len;
	int ret;
	char *name;

	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("cannot initialize libpfm\n");

	if (pfms_initialize())
		fatal_error("cannot initialize libpfms\n");

	pfm_get_num_counters(&num_counters);
	pfm_get_max_event_name_len(&len);

	name = malloc(len+1);
	if (name == NULL)
		fatal_error("cannot allocate memory for event name\n");

	memset(&ctx, 0, sizeof(ctx));
	memset(pc, 0, sizeof(pc));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	cpu_list = argc > 1 ? strtoul(argv[1], NULL, 0) : 0x3;

	ncpus = popcount(cpu_list);

		if (pfm_get_cycle_event(&inp.pfp_events[0].event) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1].event) != PFMLIB_SUCCESS)
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
	dprint("ncpus=%u npmds=%u\n", ncpus, npmds);

	pd = calloc(npmds, sizeof(pfarg_pmd_t));
	if (pd == NULL)
		fatal_error("cannot allocate pd array\n");

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	for(l=0, k = 0; l < ncpus; l++) {
		for (i=0, j=0; i < inp.pfp_event_count; i++, k++) {
			pd[k].reg_num   = outp.pfp_pmcs[j].reg_pmd_num;
			for(; j < outp.pfp_pmc_count; j++)  if (outp.pfp_pmcs[j].reg_evt_idx != i) break;
		}
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
	ret = pfms_write_pmds(desc, pd, inp.pfp_event_count);
	if (ret == -1)
		fatal_error("write_pmds error %d\n", ret);

	/*
	 * load context on all CPUs of interest
	 */
	ret = pfms_load(desc);
	if (ret == -1)
		fatal_error("load error %d\n", ret);

	/*
	 * start monitoring on all CPUs of interest
	 */
	ret = pfms_start(desc);
	if (ret == -1)
		fatal_error("start error %d\n", ret);

	/*
	 * simulate some work
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
	 * 	- the first 2 elements of pd[] read on 1st CPU
	 * 	- the next  2 elements of pd[] read on the 2nd CPU
	 * 	- and so on
	 */
	ret = pfms_read_pmds(desc, pd, npmds);
	if (ret == -1)
		fatal_error("read_pmds error %d\n", ret);

	/*
	 * pre per-CPU results
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
#endif
