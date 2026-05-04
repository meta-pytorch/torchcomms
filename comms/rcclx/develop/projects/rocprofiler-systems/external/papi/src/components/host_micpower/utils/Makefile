CC = gcc
CFLAGS = -O2 -Wall
LFLAGS = 
PAPI_INCLUDE = ../../..
PAPI_LIBRARY = ../../../libpapi.a

all:	host_micpower_plot

host_micpower_plot:	host_micpower_plot.o
	$(CC) $(LFLAGS) -o host_micpower_plot host_micpower_plot.o $(PAPI_LIBRARY) -ldl -lpthread

host_micpower_plot.o:	host_micpower_plot.c
	$(CC) $(CFLAGS) -I$(PAPI_INCLUDE) -c host_micpower_plot.c

clean:	
	rm -f *~ *.o host_micpower_plot results.*
