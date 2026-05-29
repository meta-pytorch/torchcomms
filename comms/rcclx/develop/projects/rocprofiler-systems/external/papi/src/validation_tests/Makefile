# File: validation_tests/Makefile

include Makefile.target

INCLUDE = -I../testlib -I.. -I.
testlibdir= ../testlib
TESTLIB= $(testlibdir)/libtestlib.a
DOLOOPS= $(testlibdir)/do_loops.o
CLOCKCORE= $(testlibdir)/clockcore.o
EXTRALIB = -lrt

include Makefile.recipies

.PHONY : install

install:	all
	@echo "Validation tests (DATADIR) being installed in: \"$(DATADIR)\"";
	-mkdir -p $(DATADIR)/validation_tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/validation_tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/validation_tests \;
	-chmod go+rx $(DATADIR)/validation_tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/validation_tests \;
	-cp Makefile.target $(DATADIR)/validation_tests/Makefile
	-cat Makefile.recipies >> $(DATADIR)/validation_tests/Makefile

