NAME=sensors_ppc
include ../../Makefile_comp_tests.target

TESTS = sensors_ppc_basic

sensors_ppc_tests: $(TESTS)

sensors_ppc_basic.o:	sensors_ppc_basic.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c sensors_ppc_basic.c -o sensors_ppc_basic.o

sensors_ppc_basic: sensors_ppc_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o sensors_ppc_basic sensors_ppc_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o *~
