# File: ctests/Makefile

include Makefile.target

INCLUDE = -I../testlib -I../validation_tests -I.. -I.

testlibdir= ../testlib
TESTLIB= $(testlibdir)/libtestlib.a
DOLOOPS= $(testlibdir)/do_loops.o
CLOCKCORE= $(testlibdir)/clockcore.o

validationlibdir= ../validation_tests
TESTFLOPS= $(validationlibdir)/flops_testcode.o
TESTINS= $(validationlibdir)/instructions_testcode.o
TESTCYCLES = $(validationlibdir)/busy_work.o
DISPLAYERROR= $(validationlibdir)/display_error.o

include Makefile.recipies

.PHONY : install

install: default
	@echo "C tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/ctests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/ctests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/ctests \;
	-chmod go+rx $(DATADIR)/ctests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/ctests \;
	-cp Makefile.target $(DATADIR)/ctests/Makefile
	-cat Makefile.recipies >> $(DATADIR)/ctests/Makefile
