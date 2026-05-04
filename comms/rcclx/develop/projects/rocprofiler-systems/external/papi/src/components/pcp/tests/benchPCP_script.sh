#------------------------------------------------------------------------------
# WARNING: If you use a batch job submission utility, do NOT run this script
# with it, change the line below that runs ./benchPCP to submit each instance
# of running it using the batch system. 
# Otherwise, you have a batch system running this script that is trying to 
# launch batch system jobs. On our system, that causes problems. So 
# sh benchPCP_script.sh 
# and it will do many 
# jsrun --np 1 ./benchPCP $READS $EVENT_NAME etc.
#
# Change the variables below to set the event to test, the file name, and the
# total number of samples (COUNT) and number of reads to average (READS).
# 
# benchPCP is a way to produce statistics on how long it takes to initialize
# PAPI with the PCP component, and how long individual reads take. Each 
# execution of benchPCP produces 3 measurements: These are:
# Initialize:              uS required for PAPI_library_init(). 
# us 100 PCP Event Reads:  uS required to read the EVENT_NAME event 100 times.
#
# The point of this script is to execute benchPCP many times to produce a CSV
# file, and then execute the benchStat program to produce summary statistics
# on that collection to report to the screen. The CSV file can be read by a
# spreadsheet, to produce graphs or other 
#------------------------------------------------------------------------------
# To get names, go to papi/src/utils/ and run papi_native_avail; this shows
# all the names you can compare. You do not have to put the 'pcp:::' prefix on
# PCP names, it works with or without that. 
#
# PCP_NAME="pmcd.openfds"
# PCP_NAME="pcp:::mem.util.available"
#------------------------------------------------------------------------------

COUNT=500 # Number of experiments to do.
READS=100 # Number of times to read the value.
EVENT_NAME="perfevent.active"
# EVENT_NAME="PAPI_TOT_INS"
OUTFILE="benchPCP_test_perfevent-active.csv"

# First tell benchPCP to output a header (no args).
jsrun --np 1 ./benchPCP >$OUTFILE

# Execute 'COUNT' experiments.
COUNTER=0
   while [  $COUNTER -lt $COUNT ]; do
      jsrun --np 1 ./benchPCP $READS $EVENT_NAME 2>/dev/null >>$OUTFILE
      let COUNTER=COUNTER+1 
      echo Finished: $COUNTER
   done
