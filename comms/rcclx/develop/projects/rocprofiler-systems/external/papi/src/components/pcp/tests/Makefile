NAME=pcp
include ../../Makefile_comp_tests.target

TESTS = testPCP

pcp_tests: $(TESTS)

testPCP.o:	testPCP.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c testPCP.c -o testPCP.o -fverbose-asm -Wa,-adhln=testPCP.s

testPCP: testPCP.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o testPCP testPCP.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -Xlinker -Map=testPCP_link.map

clean:
	rm -f $(TESTS) *.o *~
