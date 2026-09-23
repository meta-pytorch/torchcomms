#!/bin/sh
#
#	Transform the papi_events.csv file into a static table.
#
#	tr "\r" "\n" |		# convert CR to LF
#	tr -s "\n" |		# convert LFLF to LF
#	tr "\"" "'" |		# convert " to '
#	sed 's/^/"/' | \	# insert " at beginning of line
#	sed 's/$/\\n\"/'	# insert LF" at end of line
#
# print "#define STATIC_PAPI_EVENTS_TABLE 1"
echo "static char *papi_events_table ="
cat $1 | \
	tr "\r" "\n" |
	tr -s "\n" |
	tr "\"" "'" |
	sed 's/^/"/' | \
	sed 's/$/\\n\"/'
echo ";"
