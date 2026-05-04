//-----------------------------------------------------------------------------
// There are four files involved in bench testing. These are 
// Makefile2                     To build executables benchPCP and benchTest. 
// benchPCP_script.sh            To run benchPCP repeatedly, and output a file
//                               of many runs for statistical analysis.
// benchPCP.c                    A program that times the reading of events.
// benchStats.c                  A program to compose statistics on the output
//                               of benchPCP_script.h and output them. 
//
// Usage: 
// You must have installed PAPI with PCP, and have a system with the PCP
// daemon present. 
// 
// In our test system we run on a cluster that uses job-scheduling software;
// specifically 'bsub' and 'jsrun' to request resources then submit a job to be
// run on the cluster. Users may have other methods. The largest distinction
// for us is that we can compile and run shell scripts and code without using
// jsrun; but jsrun is required to run on a node supporting PCP.
// 
// In the file papi/src/component/pcp/linux-pcp.c, we restrict the PCP events
// to a subset; but this can be modified and papi reinstalled to give greater
// access. 

//-----------------------------------------------------------------------------
// benchPCP will time an event (PCP or otherwise) and the time it takes to do
// PAPI_library_init(). Particularly for PCP, it requires a dynamic link
// library (DLL) to run; and the first time such a library is referenced the
// system takes time to load it. This time won't appear for subsequent
// accesses, even though they are different program runs or script runs (the
// DLL is cached). This extra time is evident in the first benchPCP execution,
// but only if the shell is 'fresh'.  We make no attempt to unload the DLL.
// benchPCP produces a heading if invoked with no arguments, or times the
// intialization and the reads and reports those. It takes two arguments; the
// first is an integer for how many reads to average to get an accurate
// per-read timing (100 is typical), and the second is the event name to read.  
//
// Although benchPCP reads the events, it does nothing with the results. The
// only point is to time the reads.  

//-----------------------------------------------------------------------------
// The shell script benchPCP_script.sh SHOULD BE EDITED. It contains its own
// controls internally; for example:
// COUNT=500 # Number of experiments to do.
// READS=100 # Number of times to read the value.
// EVENT_NAME="perfevent.active"
// OUTFILE="benchPCP_test_peactive.csv"
// 
// This instructs the script to execute ./benchPCP 'COUNT' times (500 above),
// and put all the output if these runs into 'OUTFILE', which will be a CSV
// formatted file, with a title row. Each data line of this file will hold
// two timing values, in microseconds (uS): The time required to initialize
// the PAPI system (and all components present), and then the average time
// required to collect a value for the event 'EVENT_NAME' (above this is 
// "perfevent.active"), 'READS' times in a row (above this is 100).
//
// Thus the following plan:
// 1) start microsecond timer,
// 2) for (i=0; i<READS; i++) { PAPI_read(EVENT_NAME, &value); }
// 3) stop microsecond timer.
// 4) compute and report (timeElapsed / READS).
//
// An example output: 
// Initialize, Event Read Avg uS
// 177544.0,   1094.5
// 168264.0,   1099.9
// etc, 'COUNT' data lines, each a single run of benchPCP
// 
// The first line says PAPI_library_init() took 177,544 us (177.5 ms), and 
// the reads of EVENT_NAME took ON AVERAGE 1094.5 us per read. 
// 
// The user can visually inspect the file, or read it into a spreadsheet for
// statistical manipulations, sorting, or graphing. Outlier data lines can be
// deleted from the file without harm. This is the format expected by
// benchStats, the next program to be described.
//
//-----------------------------------------------------------------------------
// benchStats will produce a display of statistics on statistics the authors
// find useful in characterizing the performance of PAPI and PCP. One report
// for each column is reported.
// 
// Usage: ./benchStats filename.csv
// It requires only one argument, the output of benchPCP_script.sh.  It will
// produce the following statistics, once for each column; the following are
// redirected stdout to a file (it will not include the '// ' leader).
// 
//-----------------------------------------------------------------------------
//  
// Stats for Initialization time in file 'benchPCP_test_perfevent.csv'.
// Sample Values                  ,     500
// Minimum uS                     ,160156.0
// Maximum uS                     ,193629.0
// Average uS                     ,172860.8
// Median  uS                     ,175086.0
// First   uS                     ,177544.0
// Max w/o First                  ,193629.0
// Range   uS                     , 33473.0
// Histogram Bins chosen          ,      23
// Bin width uS                   ,  1456.0
// Mode (center highest Bin Count),178356.0
// Mode Bin Count                 ,     121
// Bin Expected Count             ,      22
// 
// Initialization Histogram:
// binCenter, Count, % of Count
// 160884.0,       30, = 6.00%
// 162340.0,       31, = 6.20%
// 163796.0,       59, =11.80%
// 165252.0,       19, = 3.80%
// 166708.0,       13, = 2.60%
// 168164.0,       12, = 2.40%
// 169620.0,       11, = 2.20%
// 171076.0,       17, = 3.40%
// 172532.0,       28, = 5.60%
// 173988.0,       25, = 5.00%
// 175444.0,       22, = 4.40%
// 176900.0,       33, = 6.60%
// 178356.0,      121, =24.20%
// 179812.0,       41, = 8.20%
// 181268.0,       21, = 4.20%
// 182724.0,        7, = 1.40%
// 184180.0,        1, = 0.20%
// 185636.0,        0, = 0.00%
// 187092.0,        2, = 0.40%
// 188548.0,        1, = 0.20%
// 190004.0,        3, = 0.60%
// 191460.0,        1, = 0.20%
// 192916.0,        2, = 0.40%
// 
// Stats for PCP event read time in file 'benchPCP_test_perfevent.csv'.
// Sample Values                  ,     500
// Minimum uS                     ,  1042.5
// Maximum uS                     ,  2365.7
// Average uS                     ,  1384.0
// Median  uS                     ,  1426.8
// First   uS                     ,  1094.5
// Max w/o First                  ,  2365.7
// Range   uS                     ,  1323.2
// Histogram Bins chosen          ,      23
// Bin width uS                   ,    58.0
// Mode (center highest Bin Count),  1419.5
// Mode Bin Count                 ,     218
// Bin Expected Count             ,      22
// 
// Read Event Histogram:
// binCenter, Count, % of Count
//   1071.5,       51, =10.20%
//   1129.5,       20, = 4.00%
//   1187.5,       21, = 4.20%
//   1245.5,        4, = 0.80%
//   1303.5,        0, = 0.00%
//   1361.5,        0, = 0.00%
//   1419.5,      218, =43.60%
//   1477.5,      184, =36.80%
//   1535.5,        0, = 0.00%
//   1593.5,        0, = 0.00%
//   1651.5,        0, = 0.00%
//   1709.5,        0, = 0.00%
//   1767.5,        0, = 0.00%
//   1825.5,        0, = 0.00%
//   1883.5,        0, = 0.00%
//   1941.5,        1, = 0.20%
//   1999.5,        0, = 0.00%
//   2057.5,        0, = 0.00%
//   2115.5,        0, = 0.00%
//   2173.5,        0, = 0.00%
//   2231.5,        0, = 0.00%
//   2289.5,        0, = 0.00%
//   2347.5,        1, = 0.20%
// 
//-----------------------------------------------------------------------------
// Discussion.
// All of these measures are prone to distortion by "weather" in the computing
// environment; meaning the particular mix of other code running, their
// priorities and use of common resources. Such noise can produce outliers,
// particularly on the high end. In general for computing environments, the
// Minimum (barring program failure) has the smallest noise component. 
// 
// Minimum and Maximum are self-explanatory.  The difference between them is
// computed as the Range (reported); a particularly wide range (e.g. more than
// half the Minimum) indicates either an extreme Maximum, or a general high
// noise environment, which may happen on a heavily used cluster.
// 
// Average and Median. These are two measures of centrality. If the
// distribution of values is not skewed, they are the same. The median is the
// most statistically robust measure, it is the 50/50 point; there is a 50%
// chance a measure falls beneath it, and 50% chance it exceeds it. 
//
// However, most distributions ARE skewed. If the Average is LESS than the
// Median, this indicates a skew left, which in turn means the left tail of the
// distribution is more drawn out than the right tail.
// 
// If the Average is MORE than the Median, this means the opposite, the left
// tail is less drawn out than the right tail. That can be caused by a few
// extreme high end samples, or even just one outlier. Unlike the median, if we
// had hundreds of values in the [1000, 1200] range, but a single outlier
// sample of 1,000,000, we could see a Median of 1100, but an average of 3100:
// Given 500 samples, that (1,000,000 / 500)=2000 added to the average all by
// itself. This is why the median is called "robust", it is completely immune
// to the magnitude of outliers.
// 
// First, Max w/o First: This lets us isolate system-wide one-time costs. In
// particular, when PAPI is configured with the PCP component, the execution of
// PAPI_library_init() must also initialize communications with PCP. This
// requires code residing in a DLL (Dynamic Link Library) provided with PCP.
// The very first time in a new shell, it takes some time for the OS to find
// and load the PCP DLL, but then it is cached and this cost is avoided for any
// subsequent PAPI_library_init() executions. However, after loading the DLL,
// there is more work to be done; the component retrieves information from PCP
// about what events it can provide and then translate and provide that
// information to PAPI, so PAPI can use them.  Because of the DLL, the very
// first Initialization cost MIGHT be many milliseconds longer than the
// subsequent Initialization costs. So we provide both the Maximum overall, and
// the Maximum without considering the first Dataline in the file, and we
// provide the first as well. The difference between the First and Maximum w/o
// First, if positive and large, can provide an estimated cost of finding,
// loading and caching the PCP DLL. If that difference is not significant, the
// DLL was likely already cached when the first benchPCP executed.
// 
// Histogram: Internally we construct a histogram to identify an empirical
// major mode. We show the numeric version of that here. 
// 
// Histogram Bins chosen: The number of bins. This is set as the square root of
// the count, rounded up to the next integer. Above, sqrt(500) ~ 22.36, so the
// number of bins is 23.0.
//
// If the distribution of times is uniform, then all the bins would have the
// same count. Since the number of bins is the sqrt(Count), the number per bin
// is Count/sqrt(Count) = sqrt(Count), also. That is in Bin Expected Count; in
// the both cases above it is one less than the number of bins due to rounding.
// 
// If we made many MORE bins, they would be narrower and the expected count
// would be lower. Too many, and it becomes difficult to see the shape of the
// distribution; eventually each sample is likely the only resident in its bin
// and we see nothing. If we made FEWER bins, we would have higher counts in
// each, but the width of bins would be wider and thus less accurate in what
// they tell us about the distribution. Again, eventually we have one bin with
// all the samples in it! Here we choose bins and counts to be nearly the same;
// that is the balancing act between accuracy (narrow bin widths) and shape,
// having enough bins to have a diversity of counts. There is no definitive
// answer to computing an ideal bin width; it is the topic of several studies.
//
// Bin width is the Range (max - min) divided by the number of bins and rounded
// up to the next integer. For the Installation histogram above, the Bin width
// is 1456 uS. For the Read Event, the Bin width is 58.0 uS.
//
// We distribute the values into the bins we created. We find the bin with the
// most values in it, and that is the Mode.  We report in "Mode Bin Count" the
// number of samples that fell into that bin, so you can compare it to the Bin
// Expected Count from above. The Mode is the center of the range, there is
// half the Bin width on either side of it. This is a third measure of
// centrality, the highest count bin is the most likely bin for such a
// measurement to land in. The closer the Mode Bin Count is to the Bin Expected
// Value, the more uniform (flatter) your distribution. You can be sure of that
// by looking at the histogram percentages.
//
// MULTI_MODAL distributions: Some histograms will have more than one peak; on
// both of the examples above, there are two peaks. If this does not go away
// with more samples, it is a sign that the computing environment has distinct
// 'modes', i.e. the environment changes periodically so the histogram
// represents a mashup of multiple normal distributions. That could be other
// user code running, OS functions running, task switching, code migration,
// swapping, paging, etc.  
//
// How to use all this info: You will pay the one-time cost of initializing PCP
// to use PCP, and the Read Event costs each time you read an event. If you are
// doing either of these many times, you should estimate that cost as one of
// the measures of centrality; the average, median, or mode; to be conservative
// take the highest of those.
