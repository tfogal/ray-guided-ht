set terminal postscript color enhanced
set notitle
set output "HashSize.eps"
set xlabel "Hash Table Size" font ",22"
set ylabel "Time (ms)" font ",22"
set xtics font ",20"
set ytics font ",20"
set autoscale

set style data linespoints
set style line 1 linecolor rgb "red" lw 5
set style line 2 linecolor rgb "blue" lw 5
#set style line 1 lt 2 lw 8 pt 3 ps 0.5

plot \
	"ht-size-naive.data" using 1:2 smooth unique title "Naive" ls 1 with lines, \
	"ht-size-staging.data" using 1:2 smooth unique title "Staging" with lines ls 2
