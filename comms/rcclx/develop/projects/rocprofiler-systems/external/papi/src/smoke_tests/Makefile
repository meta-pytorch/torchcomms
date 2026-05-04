
EXECUTABLES = simple threads

CC ?= gcc

PAPI_ROOT := $(shell dirname $(shell dirname $(shell which papi_component_avail)))
CFLAGS ?= -O0 -pthread -I$(PAPI_ROOT)/include
CPPFLAGS = -I$(PAPI_ROOT)/include
LIBS = -L$(PAPI_ROOT)/lib  -lm -ldl -lpapi -Wl,-rpath=$(PAPI_ROOT)/lib -pthread

all: $(EXECUTABLES)

clean:
	/bin/rm -f core *.o $(EXECUTABLES)

i.SUFFIXES: .c .o
	.c.o:
	    $(CC) $(CFLAGS) -c $*.c

simple: simple.o 
	$(CC) $(CFLAGS) -o simple simple.o $(LIBS)
threads: threads.o 
	$(CC) $(CFLAGS) -o threads threads.o $(LIBS)
