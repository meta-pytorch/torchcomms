#!/bin/sh

# File:    run_example.sh
# CVS:     $Id$
# Author:  Min Zhou
#          min@cs.utk.edu


CTESTS=`find . -perm -u+x -type f`;
ALLTESTS="$CTESTS";
x=0;
CWD=`pwd`

echo "Platform:"
uname -a

echo ""
echo "The following test cases will be run:";
echo $ALLTESTS;

echo "";
echo "Running C Example Programs";
echo ""

for i in $CTESTS;
do
if [ -x $i ]; then
if [ "$i" != "./run_examples.sh" ]; then
echo  "Running $i: ";
./$i 
fi;
fi;
echo "";
done

