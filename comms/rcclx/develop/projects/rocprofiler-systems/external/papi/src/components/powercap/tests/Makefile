NAME=powercap
include ../../Makefile_comp_tests.target

TESTS = powercap_basic powercap_limit powercap_basic_read powercap_basic_readwrite

powercap_tests: $(TESTS)

powercap_basic.o: powercap_basic.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c powercap_basic.c -o powercap_basic.o -DBASIC_TEST=1

powercap_basic: powercap_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o powercap_basic powercap_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


powercap_limit.o: powercap_limit.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c powercap_limit.c -o powercap_limit.o

powercap_limit: powercap_limit.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o powercap_limit powercap_limit.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)


powercap_basic_read.o: powercap_basic_read.c
	$(CC) $(CFLAGS) -Wall $(OPTFLAGS) $(INCLUDE) -c powercap_basic_read.c -o powercap_basic_read.o -DBASIC_TEST=1

powercap_basic_read: powercap_basic_read.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o powercap_basic_read powercap_basic_read.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


powercap_basic_readwrite.o: powercap_basic_readwrite.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c powercap_basic_readwrite.c -o powercap_basic_readwrite.o -DBASIC_TEST=1

powercap_basic_readwrite: powercap_basic_readwrite.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o powercap_basic_readwrite powercap_basic_readwrite.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o *~
