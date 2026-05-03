NAME=appio
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = appio_list_events appio_values_by_code appio_values_by_name appio_test_read_write appio_test_pthreads appio_test_fread_fwrite appio_test_seek

ALL_TESTS = $(TESTS) appio_test_blocking appio_test_select appio_test_recv appio_test_socket

appio_tests: $(TESTS)

all: $(ALL_TESTS)

ARCH=$(shell uname -m)

ifeq (x86_64,$(ARCH))
        ARCH_SUFFIX="-AMD64"
endif


%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

appio_list_events: appio_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_list_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_values_by_code: appio_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_values_by_code.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_values_by_name: appio_values_by_name.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_values_by_name.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_read_write: appio_test_read_write.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_read_write.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_seek: appio_test_seek.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_seek.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_blocking: appio_test_blocking.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_blocking.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_socket: appio_test_socket.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_socket.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_recv: appio_test_recv.o $(UTILOBJS) ../../../libpapi.so
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_recv.o $(UTILOBJS) -Wl,-rpath ../../..  ../../../libpapi.so $(LDFLAGS)

appio_test_select: appio_test_select.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_select.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_fread_fwrite: appio_test_fread_fwrite.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_fread_fwrite.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_pthreads: appio_test_pthreads.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_pthreads.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lpthread

init_fini.o: init_fini.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $^

clean:
	rm -f $(ALL_TESTS) *.o

