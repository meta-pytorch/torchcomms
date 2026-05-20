#!/bin/sh 
# this is the configuration that goes into a fedora rpm
#./configure --with-debug --with-components="coretemp example infiniband lustre mx net" $1 
if [ -f components/cuda/Makefile.cuda ]; then
	if [ -f components/nvml/Makefile.nvml ]; then
		./configure --with-components="appio coretemp example lustre micpower mx net rapl stealtime cuda nvml" $1
	else
		./configure --with-components="appio coretemp example lustre micpower mx net rapl stealtime cuda" $1
	fi
else
	./configure --with-components="appio coretemp example lustre micpower mx net rapl stealtime" $1
fi
