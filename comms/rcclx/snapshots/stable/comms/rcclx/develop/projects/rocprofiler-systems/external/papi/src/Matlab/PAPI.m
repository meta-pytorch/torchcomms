% PAPI Performance API.
%    PAPI provides access to one of 8 Hardware Performance Monitoring functions.
%
%          ctrs = PAPI('num')   - Return the number of hardware counters.
%                 PAPI('start', 'event', ...) - 
%                                 Begin counting the specified events.
%    [val, ...] = PAPI('stop')  - Stop counting and return the current values.
%    [val, ...] = PAPI('read')  - Read the current values of the active counters.
%    [val, ...] = PAPI('accum') - Add the current values of the active counters 
%                                 to the input values.
%                 PAPI('ipc')   - Begin counting instructions.
%           ins = PAPI('ipc')   - Return the number of instructions executed
%                                 since the last call.
%    [ins, ipc] = PAPI('ipc')   - Return both the total number of instructions 
%                                 executed since the last call, and the 
%                                 incremental rate of instruction execution
%                                 since the last call.
%                 PAPI('flips')
%                 PAPI('flops') - Begin counting floating point 
%                                 instructions or operations.
%           ins = PAPI('flips')
%           ops = PAPI('flops') - Return the number of floating point instruc-
%                                 tions or operations since the last call.
%      [ins, mflips] = PAPI('flips')
%      [ops, mflops] = PAPI('flops') - 
%                                 Return both the number of floating point 
%                                 instructions or operations since the last  
%                                 call, and the incremental rate of floating  
%                                 point execution since since the last call.
%
%    DESCRIPTION
%    The PAPI function provides access to the PAPI Performance API.
%    PAPI takes advantage of the fact that most modern microprocessors
%    have built-in hardware support for counting a variety of basic operations
%    or events. PAPI uses these counters to track things like instructions 
%    executed, cycles elapsed, floating point instructions performed and 
%    a variety of other events.
%
%    There are 8 subfunctions within the PAPI call, as described below:
%      'num'   - provides information on the number of hardware counters built  
%                into this platform. The result of this call specifies how many  
%                events can be counted at once.
%      'start' - programs the counters with the named events and begins 
%                counting. The names of the events can be found in the PAPI
%                documentation. If a named event cannot be found, or cannot
%                be mapped, an error message is displayed.
%      'stop'  - stops counting and returns the values of the counters in the
%                same order as events were specified in the start command. 
%                'stop' also can be used to reset the counters for the ipc
%                flips and flops subfunctions described below.
%      'read'  - return the values of the counters without stopping them.
%      'accum' - adds the values of the counters to the input parameters and
%                returns them in the output parameters. Counting is not stopped.
%      'ipc'   - returns the total instructions executed since the first call 
%                to this subfunction, and the rate of execution of instructions  
%                (as instructions per cycle) since the last call.
%      'flips' - returns the total floating point instructions executed since 
%                the last call to this subfunction, and the rate of execution 
%                of floating point instructions (as mega-floating point
%                instructions per second, or mflips) since the last call.
%                A floating point instruction is defined as whatever this cpu
%                naturally counts as floating point instructions.
%      'flops' - identical to 'flips', except it measures floating point
%                operations rather than instructions. In many cases these two
%                counts may be identical. In some cases 'flops' will be a 
%                derived value that attempts to reproduce that which is
%                traditionally considered a floating point operation. For
%                example, a fused multiply-add would be counted as two
%                operations, even if it was only a single instruction.
%
%    In typical usage, the first five subfunctions: 'num', 'start', 'stop',
%    'read', and 'accum' are used together. 'num establishes the maximum number
%    of events that can be supplied to 'start'. After a 'start' is issued, 
%    'read' and 'accum' can be intermixed until a 'stop' is issued.
%
%    The three rate calls, 'ipc', 'flips', and 'flops' are intended to be used
%    independently. They cannot be mixed, because they use the same counter
%    resources. They can be used serially if they are separated by a 'stop'
%    call, which can also be used to reset the counters.
%
%   Copyright 2001 - 2004 The Innovative Computing Laboratory,
%                               University of Tennessee.
%   $Revision$  $Date$

