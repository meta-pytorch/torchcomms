NAME=infiniband
include ../../Makefile_comp_tests.target

TESTS = infiniband_list_events infiniband_values_by_code

ifneq ($(MPICC),)
TESTS += MPI_test_infiniband_events
endif

infiniband_tests: $(TESTS)

MPI_test_infiniband_events.o:MPI_test_infiniband_events.c
	$(MPICC) $(INCLUDE) -c -o $@ $<

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

infiniband_list_events: infiniband_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

infiniband_values_by_code: infiniband_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

MPI_test_infiniband_events: MPI_test_infiniband_events.o $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(INCLUDE) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o

