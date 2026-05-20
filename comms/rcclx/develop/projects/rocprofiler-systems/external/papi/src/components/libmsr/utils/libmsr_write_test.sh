#!/bin/bash

gpconfig="set grid; set key top left; set style line 1 lw 2 pt 3 ps 1.5; set style line 2 lw 1.5; set style line 3 lw 1.5;"
set_term_postscript="set terminal postscript eps enhanced color font 'Helvetica,19' linewidth 1.5"

ls_1=1
ls_2=2
ls_3=3
ls_4=4
ls_5=5

rm -f libmsr_write_test_output_tmp_set.dat
rm -f libmsr_write_test_output_tmp_read.dat
cat libmsr_write_test_output.txt | grep SET > libmsr_write_test_output_tmp_set.dat
cat libmsr_write_test_output.txt | grep READ > libmsr_write_test_output_tmp_read.dat

graph='plot-libmsr'
gnuplot << EOF
set title "Using PAPI libmsr component to read and set power caps\n 2x8 cores Xeon E5-2690 SandyBridge-EP at 2.9GHz"  #[scooter]
${gpconfig}
set xlabel "Elapsed time (seconds)"
set ylabel "Watts"
set yrange [0:]
set y2range [0:]
set y2label "Unit Work Time (seconds)
set xrange [0:]
set key bottom right
plot \
    'libmsr_write_test_output_tmp_read.dat' index 0 using (\$2):(\$6) title "Power Consumpution (watts)" smooth unique with linespoints ls ${ls_2}, \
    'libmsr_write_test_output_tmp_read.dat' index 0 using (\$2):(\$3) axes x1y2 title "Time for Unit Work (seconds on y2 axis)" smooth unique with linespoints ls ${ls_3},\
    'libmsr_write_test_output_tmp_set.dat' index 0 using (\$2):(\$9) title "Set Avg Power Cap (watts in 1 sec)" smooth unique with points ls ${ls_1}



${gp_pause}
set terminal jpeg;  set output "${graph}.jpg"; replot;
#${set_term_postscript}; set output "${graph}.eps"; replot;
print 'Saving files ${graph}'
EOF

#( epstopdf ${graph}.eps; rm ${graph}.eps )

