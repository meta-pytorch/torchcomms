NAME=template
include ../../Makefile_comp_tests.target

CFLAGS    = $(OPTFLAGS)
CPPFLAGS += $(INCLUDE)
LDFLAGS  += $(PAPILIB) $(TESTLIB) $(UTILOBJS)

TESTS = simple
template_tests: $(TESTS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPTFLAGS) -c -o $@ $<

simple: simple.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
