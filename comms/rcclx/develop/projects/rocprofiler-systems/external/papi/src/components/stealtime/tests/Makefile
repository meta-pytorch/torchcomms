NAME=stealtime
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = stealtime_basic

stealtime_tests: $(TESTS)

stealtime_basic: stealtime_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o stealtime_basic stealtime_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o





