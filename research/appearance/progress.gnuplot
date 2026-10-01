set terminal svg size 1440,580 font 'Arial,12' background rgb '#f5f4ef'
set output 'progress.svg'
set multiplot layout 1,2 title 'Frozen eight-asset appearance pilot: reduction versus bake work' font ',19'
set border 3 lc rgb '#b8c3bd'
set tics nomirror textcolor rgb '#48544e'
set grid ytics lc rgb '#dde2dd'
set key bottom left
set xlabel 'Summed elapsed bake seconds (shared workstation)'
set ylabel 'Category-balanced chain triangles retained (%)'
set xrange [0:*]
plot 'progress-baseline.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#7c8581' title 'Matched reference', \
'progress-original.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#aaaaaa' title 'Original incumbent', \
'progress-screen.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#bc954e' title 'Conservative screen', \
'progress-ordering.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#157f69' title 'Appearance ordering', \
'progress-attributes.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#376cbb' title 'Wedge fitting'
set xlabel 'Scheduled screen diameter (pixels, logarithmic axis)'
set ylabel 'Triangles retained per scheduled level (%)'
set logscale x
set xrange [512:16]
set yrange [0:105]
set xtics (512,256,128,64,32,16)
set key top right
plot 'levels-baseline.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#7c8581' title 'Matched reference', \
'levels-original.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#aaaaaa' title 'Original incumbent', \
'levels-screen.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#bc954e' title 'Conservative screen', \
'levels-ordering.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#157f69' title 'Appearance ordering', \
'levels-attributes.dat' using 1:2 with linespoints lw 2 pt 7 ps .8 lc rgb '#376cbb' title 'Wedge fitting'
unset multiplot
