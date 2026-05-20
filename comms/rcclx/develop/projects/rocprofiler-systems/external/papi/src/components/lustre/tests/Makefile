NAME=lustre
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = lustre_basic

lustre_tests: $(TESTS)

lustre_basic: lustre_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o lustre_basic lustre_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o





