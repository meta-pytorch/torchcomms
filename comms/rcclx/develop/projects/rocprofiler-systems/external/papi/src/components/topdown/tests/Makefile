NAME=topdown
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = topdown_basic topdown_L1 topdown_L2

topdown_tests: $(TESTS)

topdown_basic: topdown_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o topdown_basic topdown_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

topdown_L1: topdown_L1.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o topdown_L1 topdown_L1.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

topdown_L2: topdown_L2.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o topdown_L2 topdown_L2.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


clean:
	rm -f $(TESTS) *.o
