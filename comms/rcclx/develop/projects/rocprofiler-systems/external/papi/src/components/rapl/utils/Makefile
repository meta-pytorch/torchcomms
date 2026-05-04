CC = gcc
CFLAGS = -O2 -Wall
LFLAGS = 
PAPI_INCLUDE = ../../..
PAPI_LIBRARY = ../../../libpapi.a

all:	rapl_plot

rapl_plot:	rapl_plot.o
	$(CC) $(LFLAGS) -o rapl_plot rapl_plot.o $(PAPI_LIBRARY)

rapl_plot.o:	rapl_plot.c
	$(CC) $(CFLAGS) -I$(PAPI_INCLUDE) -c rapl_plot.c

clean:	
	rm -f *~ *.o rapl_plot results.*
