function PAPIInnerProduct

% Compute an Inner Product (c = a * x) 
% on elements sized from 50 to 500,
% in steps of 50.
%
% Use the PAPI mex function with two different methods:
% - The PAPI flops call
% - PAPI start/stop calls
%
% For each size, display:
% - number of floating point operations
% - theoretical number of operations
% - difference
% - per cent error
% - mflops/s

fprintf(1,'\n\nPAPI Inner Product Test');
fprintf(1,'\nUsing the PAPI("flops") call');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n', 'difference', '% error', 'mflops')
for n=50:50:500,
    a=rand(1,n);x=rand(n,1);
    PAPI('stop'); % reset the counters to zero
    PAPI('flops'); % start counting flops
    c=a*x;
    [ops, mflops] = PAPI('flops'); % read the flops data
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.2f\n',n,ops,2*n,ops - 2*n, (1.0 - ((2*n) / ops)) * 100,mflops)
end
PAPI('stop');

fprintf(1,'\n\nPAPI Inner Product Test');
fprintf(1,'\nUsing PAPI start and stop');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n', 'difference', '% error', 'flops/cycle')
for n=50:50:500,
    a=rand(1,n);x=rand(n,1);
    PAPI('start', 'PAPI_TOT_CYC', 'PAPI_FP_OPS');
    c=a*x;
    [cyc, ops] = PAPI('stop');
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.6f\n',n,ops,2*n,ops - 2*n, (1.0 - ((2*n) / ops)) * 100,ops/cyc)
end