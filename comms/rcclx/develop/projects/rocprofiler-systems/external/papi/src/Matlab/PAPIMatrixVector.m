function PAPIMatrixVector

% Compute a Matrix Vector multiply 
% on arrays and vectors sized from 50 to 500,
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

fprintf(1,'\nPAPI Matrix Vector Multiply Test');
fprintf(1,'\nUsing the PAPI("flops") call');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n^2', 'difference', '% error', 'mflops')
for n=50:50:500,
    a=rand(n);x=rand(n,1);
    PAPI('stop'); % reset the counters to zero
    PAPI('flops'); % start counting flops
    b=a*x;
    [count, mflops] = PAPI('flops'); % read the flops data
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.2f\n',n,count,2*n^2,count - 2*n^2, (1.0 - ((2*n^2) / count)) * 100,mflops)
end
PAPI('stop');

fprintf(1,'\nPAPI Matrix Vector Multiply Test');
fprintf(1,'\nUsing PAPI start and stop');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n^2', 'difference', '% error', 'flops/cycle')
for n=50:50:500,
    a=rand(n);x=rand(n,1);
    PAPI('start', 'PAPI_TOT_CYC', 'PAPI_FP_OPS');
    c=a*x;
    [cyc, ops] = PAPI('stop');
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.6f\n',n,ops,2*n^2,ops - 2*n^2, (1.0 - ((2*n^2) / ops)) * 100,ops/cyc)
end