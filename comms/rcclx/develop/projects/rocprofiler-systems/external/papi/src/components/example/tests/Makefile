NAME=example
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = example_basic example_multiple_components

example_tests: $(TESTS)

example_basic: example_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o example_basic example_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

example_multiple_components: example_multiple_components.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o example_multiple_components example_multiple_components.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o
