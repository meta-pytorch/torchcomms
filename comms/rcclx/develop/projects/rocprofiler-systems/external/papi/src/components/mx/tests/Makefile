NAME=mx
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = mx_basic mx_elapsed

mx_tests: $(TESTS)

mx_basic: mx_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o mx_basic mx_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


mx_elapsed: mx_elapsed.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o mx_elapsed mx_elapsed.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


clean:
	rm -f $(TESTS) *.o





