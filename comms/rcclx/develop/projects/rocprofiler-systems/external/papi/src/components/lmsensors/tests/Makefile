NAME=lmsensors
include ../../Makefile_comp_tests.target

TESTS = lmsensors_list_events \
        lmsensors_read

lmsensors_tests: $(TESTS)

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

lmsensors_list_events: lmsensors_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ lmsensors_list_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

lmsensors_read: lmsensors_read.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ lmsensors_read.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
