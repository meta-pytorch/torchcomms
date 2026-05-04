NAME=perf_event_uncore
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = perf_event_uncore perf_event_uncore_attach perf_event_uncore_multiple \
	perf_event_amd_northbridge perf_event_uncore_cbox

DOLOOPS= $(testlibdir)/do_loops.o

perf_event_uncore_tests: $(TESTS)


perf_event_uncore_lib.o:	perf_event_uncore_lib.c perf_event_uncore_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_uncore_lib.c


perf_event_amd_northbridge:	perf_event_amd_northbridge.o $(DOLOOPS) $(UTILOBJS) $(PAPILIB) $(DOLOOPS)
	$(CC) $(CFLAGS) -o perf_event_amd_northbridge perf_event_amd_northbridge.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) $(LDFLAGS)

perf_event_uncore:	perf_event_uncore.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) perf_event_uncore_lib.o
	$(CC) $(CFLAGS) -o perf_event_uncore perf_event_uncore.o perf_event_uncore_lib.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) $(LDFLAGS)

perf_event_uncore_attach:	perf_event_uncore_attach.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) perf_event_uncore_lib.o
	$(CC) $(CFLAGS) -o perf_event_uncore_attach perf_event_uncore_attach.o perf_event_uncore_lib.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) $(LDFLAGS)

perf_event_uncore_multiple:	perf_event_uncore_multiple.o perf_event_uncore_lib.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore_multiple perf_event_uncore_multiple.o perf_event_uncore_lib.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) $(LDFLAGS) 

perf_event_uncore_cbox:	perf_event_uncore_cbox.o perf_event_uncore_lib.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore_cbox perf_event_uncore_cbox.o perf_event_uncore_lib.o $(UTILOBJS) $(DOLOOPS) $(PAPILIB) $(LDFLAGS)



clean:
	rm -f $(TESTS) *.o *~

