/*****************************************************************/
/********* Begin perf_event low-level code ***********************/
/*****************************************************************/

/* In case headers aren't new enough to have __NR_perf_event_open */
#ifndef __NR_perf_event_open

#ifdef __powerpc__
#define __NR_perf_event_open	319
#elif defined(__x86_64__)
#define __NR_perf_event_open	298
#elif defined(__i386__)
#define __NR_perf_event_open	336
#elif defined(__arm__)
#define __NR_perf_event_open	364
#endif

#endif

static long
sys_perf_event_open( struct perf_event_attr *hw_event,
		pid_t pid, int cpu, int group_fd, unsigned long flags )
{
	int ret;

	ret = syscall( __NR_perf_event_open,
			hw_event, pid, cpu, group_fd, flags );

	return ret;
}


/*
 * We define u64 as uint64_t for every architecture
 * so that we can print it with "%"PRIx64 without getting warnings.
 *
 * typedef __u64 u64;
 * typedef __s64 s64;
 */
typedef uint64_t u64;
typedef int64_t s64;

typedef __u32 u32;
typedef __s32 s32;

typedef __u16 u16;
typedef __s16 s16;

typedef __u8  u8;
typedef __s8  s8;


#ifdef __SIZEOF_INT128__
static inline u64 mul_u64_u32_shr(u64 a, u32 b, unsigned int shift)
{
	return (u64)(((unsigned __int128)a * b) >> shift);
}

#else

#ifdef __i386__
static inline u64 mul_u32_u32(u32 a, u32 b)
{
	u32 high, low;

	asm ("mull %[b]" : "=a" (low), "=d" (high)
			 : [a] "a" (a), [b] "rm" (b) );

	return low | ((u64)high) << 32;
}
#else
static inline u64 mul_u32_u32(u32 a, u32 b)
{
	return (u64)a * b;
}
#endif

static inline u64 mul_u64_u32_shr(u64 a, u32 b, unsigned int shift)
{
	u32 ah, al;
	u64 ret;

	al = a;
	ah = a >> 32;

	ret = mul_u32_u32(al, b) >> shift;
	if (ah)
		ret += mul_u32_u32(ah, b) << (32 - shift);

	return ret;
}

#endif	/* __SIZEOF_INT128__ */

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


#if defined(__x86_64__) || defined(__i386__)


static inline unsigned long long rdtsc(void) {

	unsigned a,d;

	__asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

	return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

static inline unsigned long long rdpmc(unsigned int counter) {

	unsigned int low, high;

	__asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));

	return (unsigned long long)low | ((unsigned long long)high) <<32;
}

#define barrier() __asm__ volatile("" ::: "memory")


#elif defined(__aarch64__)

/* Indirect stringification.  Doing two levels allows the parameter to be a
 * macro itself.  For example, compile with -DFOO=bar, __stringify(FOO)
 * converts to "bar".
 */

#define __stringify_1(x...)     #x
#define __stringify(x...)       __stringify_1(x)

#define read_sysreg(r) ({						\
	u64 __val;							\
	asm volatile("mrs %0, " __stringify(r) : "=r" (__val));		\
	__val;								\
})

static u64 read_pmccntr(void)
{
	return read_sysreg(pmccntr_el0);
}

#define PMEVCNTR_READ(idx)					\
	static u64 read_pmevcntr_##idx(void) {			\
		return read_sysreg(pmevcntr##idx##_el0);	\
	}

PMEVCNTR_READ(0);
PMEVCNTR_READ(1);
PMEVCNTR_READ(2);
PMEVCNTR_READ(3);
PMEVCNTR_READ(4);
PMEVCNTR_READ(5);
PMEVCNTR_READ(6);
PMEVCNTR_READ(7);
PMEVCNTR_READ(8);
PMEVCNTR_READ(9);
PMEVCNTR_READ(10);
PMEVCNTR_READ(11);
PMEVCNTR_READ(12);
PMEVCNTR_READ(13);
PMEVCNTR_READ(14);
PMEVCNTR_READ(15);
PMEVCNTR_READ(16);
PMEVCNTR_READ(17);
PMEVCNTR_READ(18);
PMEVCNTR_READ(19);
PMEVCNTR_READ(20);
PMEVCNTR_READ(21);
PMEVCNTR_READ(22);
PMEVCNTR_READ(23);
PMEVCNTR_READ(24);
PMEVCNTR_READ(25);
PMEVCNTR_READ(26);
PMEVCNTR_READ(27);
PMEVCNTR_READ(28);
PMEVCNTR_READ(29);
PMEVCNTR_READ(30);

/*
 * Read a value direct from PMEVCNTR<idx>
 */
static u64 rdpmc(unsigned int counter)
{
	static u64 (* const read_f[])(void) = {
		read_pmevcntr_0,
		read_pmevcntr_1,
		read_pmevcntr_2,
		read_pmevcntr_3,
		read_pmevcntr_4,
		read_pmevcntr_5,
		read_pmevcntr_6,
		read_pmevcntr_7,
		read_pmevcntr_8,
		read_pmevcntr_9,
		read_pmevcntr_10,
		read_pmevcntr_11,
		read_pmevcntr_13,
		read_pmevcntr_12,
		read_pmevcntr_14,
		read_pmevcntr_15,
		read_pmevcntr_16,
		read_pmevcntr_17,
		read_pmevcntr_18,
		read_pmevcntr_19,
		read_pmevcntr_20,
		read_pmevcntr_21,
		read_pmevcntr_22,
		read_pmevcntr_23,
		read_pmevcntr_24,
		read_pmevcntr_25,
		read_pmevcntr_26,
		read_pmevcntr_27,
		read_pmevcntr_28,
		read_pmevcntr_29,
		read_pmevcntr_30,
		read_pmccntr
	};

	if (counter < ARRAY_SIZE(read_f))
		return (read_f[counter])();

	return 0;
}

static u64 rdtsc(void) { return read_sysreg(cntvct_el0); }

#define barrier()	asm volatile("dmb ish" : : : "memory")

#endif

#if defined(__x86_64__) || defined(__i386__) || defined(__aarch64__)

static inline u64 adjust_cap_usr_time_short(u64 a, u64 b, u64 c)
{
	u64 ret;
	ret = b + ((a - b) & c);
	return ret;
}

/* based on the code in include/uapi/linux/perf_event.h */
static inline unsigned long long mmap_read_self(void *addr,
					 int user_reset_flag,
					 unsigned long long reset,
					 unsigned long long *en,
					 unsigned long long *ru) {

	struct perf_event_mmap_page *pc = addr;

	uint32_t seq, time_mult = 0, time_shift = 0, index, width;
	int64_t count;
	uint64_t enabled, running;
	uint64_t cyc = 0, time_offset = 0, time_cycles = 0, time_mask = ~0ULL;
	int64_t pmc = 0;
	uint64_t delta = 0;


	do {
		/* The kernel increments pc->lock any time */
		/* perf_event_update_userpage() is called */
		/* So by checking now, and the end, we */
		/* can see if an update happened while we */
		/* were trying to read things, and re-try */
		/* if something changed */
		/* The barrier ensures we get the most up to date */
		/* version of the pc->lock variable */

		seq=pc->lock;
		barrier();

		/* For multiplexing */
		/* time_enabled is time the event was enabled */
		enabled = pc->time_enabled;
		/* time_running is time the event was actually running */
		running = pc->time_running;

		/* if cap_user_time is set, we can use rdtsc */
		/* to calculate more exact enabled/running time */
		/* for more accurate multiplex calculations */
		if ( (pc->cap_user_time) && (enabled != running)) {
			cyc = rdtsc();
			time_offset = pc->time_offset;
			time_mult = pc->time_mult;
			time_shift = pc->time_shift;

			if (pc->cap_user_time_short) {
				time_cycles = pc->time_cycles;
				time_mask = pc->time_mask;
			}
		}

		/* actually do the measurement */

		/* Index of register to read */
		/* 0 means stopped/not-active */
		/* Need to subtract 1 to get actual index to rdpmc() */
		index = pc->index;

		/* count is the value of the counter the last time */
		/* the kernel read it */
		/* If we don't sign extend it, we get large negative */
		/* numbers which break if an IOC_RESET is done */
		width = pc->pmc_width;
		count = pc->offset;
		if (user_reset_flag == 1) {
			count = 0;
		}

		/* Ugh, libpfm4 perf_event.h has cap_usr_rdpmc */
		/* while actual perf_event.h has cap_user_rdpmc */

		/* Only read if rdpmc enabled and event index valid */
		/* Otherwise return the older (out of date?) count value */
		if (pc->cap_usr_rdpmc && index) {

			/* Read counter value */
			pmc = rdpmc(index-1);

			/* sign extend result */
			if (user_reset_flag == 1) {
				pmc-=reset;
			}
			pmc<<=(64-width);
			pmc>>=(64-width);

			/* add current count into the existing kernel count */
			count+=pmc;
		} else {
			/* Falling back because rdpmc not supported	*/
			/* for this event.				*/
			return 0xffffffffffffffffULL;
		}

		barrier();

	} while (pc->lock != seq);

	if (enabled != running) {

		/* Adjust for cap_usr_time_short, a nop if not */
		cyc = adjust_cap_usr_time_short(cyc, time_cycles, time_mask);

		delta = time_offset + mul_u64_u32_shr(cyc, time_mult, time_shift);

		enabled+=delta;
		if (index)
			/* Only adjust if index is valid */
			running+=delta;
	}

	if (en) *en=enabled;
	if (ru) *ru=running;

	return count;
}

static inline unsigned long long mmap_read_reset_count(void *addr) {

	struct perf_event_mmap_page *pc = addr;
	uint32_t seq, index;
	uint64_t count = 0;

	if (pc == NULL)  {
	return count;
	}

	do {
		/* The barrier ensures we get the most up to date */
		/* version of the pc->lock variable */

		seq=pc->lock;
		barrier();

		/* actually do the measurement */

		/* Ugh, libpfm4 perf_event.h has cap_usr_rdpmc */
		/* while actual perf_event.h has cap_user_rdpmc */

		/* Index of register to read */
		/* 0 means stopped/not-active */
		/* Need to subtract 1 to get actual index to rdpmc() */
		index = pc->index;

		if (pc->cap_usr_rdpmc && index) {
			/* Read counter value */
			count = rdpmc(index-1);
		}
		barrier();

	} while (pc->lock != seq);

	return count;
}

#else
static inline unsigned long long mmap_read_self(void *addr __attribute__((unused)),
					 int user_reset_flag __attribute__((unused)),
					 unsigned long long reset __attribute__((unused)),
					 unsigned long long *en __attribute__((unused)),
					 unsigned long long *ru __attribute__((unused))) {

	return (unsigned long long)(-1);
}

static inline unsigned long long mmap_read_reset_count(void *addr __attribute__((unused))) {

	return (unsigned long long)(-1);
}

#endif

/* These functions are based on builtin-record.c in the  */
/* kernel's tools/perf directory.                        */
/* This code is from a really ancient version of perf */
/* And should be updated/commented properly */


static uint64_t
mmap_read_head( pe_event_info_t *pe )
{
	struct perf_event_mmap_page *pc = pe->mmap_buf;
	int head;

	if ( pc == NULL ) {
		PAPIERROR( "perf_event_mmap_page is NULL" );
		return 0;
	}

	head = pc->data_head;
	rmb();

	return head;
}

static void
mmap_write_tail( pe_event_info_t *pe, uint64_t tail )
{
	struct perf_event_mmap_page *pc = pe->mmap_buf;

	/* ensure all reads are done before we write the tail out. */
	pc->data_tail = tail;
}

/* Does the kernel define these somewhere? */
struct ip_event {
	struct perf_event_header header;
 	uint64_t ip;
};
struct lost_event {
	struct perf_event_header header;
	uint64_t id;
	uint64_t lost;
};
typedef union event_union {
	struct perf_event_header header;
	struct ip_event ip;
	struct lost_event lost;
} perf_sample_event_t;

/* Should re-write with comments if we ever figure out what's */
/* going on here.                                             */
static void
mmap_read( int cidx, ThreadInfo_t **thr, pe_event_info_t *pe,
           int profile_index )
{
	uint64_t head = mmap_read_head( pe );
	uint64_t old = pe->tail;
	unsigned char *data = ((unsigned char*)pe->mmap_buf) + getpagesize();
	int diff;

	diff = head - old;
	if ( diff < 0 ) {
		SUBDBG( "WARNING: failed to keep up with mmap data. head = %" PRIu64
			",  tail = %" PRIu64 ". Discarding samples.\n", head, old );
		/* head points to a known good entry, start there. */
		old = head;
	}

	for( ; old != head; ) {
		perf_sample_event_t *event = ( perf_sample_event_t * )& data[old & pe->mask];
		perf_sample_event_t event_copy;
		size_t size = event->header.size;

		/* Event straddles the mmap boundary -- header should always */
		/* be inside due to u64 alignment of output.                 */
		if ( ( old & pe->mask ) + size != ( ( old + size ) & pe->mask ) ) {
			uint64_t offset = old;
			uint64_t len = min( sizeof ( *event ), size ), cpy;
			void *dst = &event_copy;

			do {
				cpy = min( pe->mask + 1 - ( offset & pe->mask ), len );
				memcpy( dst, &data[offset & pe->mask], cpy );
				offset += cpy;
				dst = ((unsigned char*)dst) + cpy;
				len -= cpy;
			} while ( len );

			event = &event_copy;
		}
		old += size;

		SUBDBG( "event->type = %08x\n", event->header.type );
		SUBDBG( "event->size = %d\n", event->header.size );

		switch ( event->header.type ) {
			case PERF_RECORD_SAMPLE:
				_papi_hwi_dispatch_profile( ( *thr )->running_eventset[cidx],
					( vptr_t ) ( unsigned long ) event->ip.ip,
					0, profile_index );
				break;

			case PERF_RECORD_LOST:
				SUBDBG( "Warning: because of a mmap buffer overrun, %" PRId64
					" events were lost.\n"
					"Loss was recorded when counter id %#"PRIx64
					" overflowed.\n", event->lost.lost, event->lost.id );
				break;
			default:
				SUBDBG( "Error: unexpected header type - %d\n",
					event->header.type );
				break;
		}
	}

	pe->tail = old;
	mmap_write_tail( pe, old );
}


