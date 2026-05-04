NAME=vmware
include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = vmware_basic

vmware_tests: $(TESTS)

vmware_basic: vmware_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o vmware_basic vmware_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o





