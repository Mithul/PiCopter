 #set multiplot layout 2,2 rowsfirst
 set xrange [2000:500]
 # set yrange [0.35:0.38]
 # plot 'motor.log' using 1:8 with lines title 'gx', 'motor.log' using 1:9 with lines title 'gy','motor.log' using 1:10 with lines title 'ax', 'motor.log' using 1:11 with lines title 'ay', 'motor.log' using 1:5 with lines title 'pitch', 'motor.log' using 1:6 with lines title 'roll'
 plot 'motor.log' using 1:18 with lines title 'new_pitch', 'motor.log' using 1:19 with lines title 'new_roll','motor.log' using 1:11 with lines title 'pitch', 'motor.log' using 1:12 with lines title 'roll'#, 'motor.log' using 1:7 with lines title 'rd', 'motor.log' using 1:6 with lines title 'ri', 'motor.log' using 1:5 with lines title 'rp', 'motor.log' using 1:4 with lines title 'pd', 'motor.log' using 1:3 with lines title 'pi', 'motor.log' using 1:2 with lines title 'pp'
 # plot 'motor.log' using 1:9 with lines title 'gx', 'motor.log' using 1:10 with lines title 'gy', 'motor.log' using 1:6 with lines title 'pitch', 'motor.log' using 1:7 with lines title 'roll'
 # plot 'motor.log' using 1:3 with lines title 'x-pid', 'motor.log' using 1:4 with lines title 'y-pid', 'motor.log' using 1:6 with lines title 'pitch', 'motor.log' using 1:7 with lines title 'roll'

 # plot 'motor.log' using 1:9 with lines title 'actual-time'
#  plot 'motor.log' using 1:8 with lines title 'time', 'motor.log' using 1:9 with lines title 'actual-time'

pause 0.1
reread

# print 'hello'
# stats datafile
# print STATS_blocks

# set key autotitle columnheader
# set yrange [-0.3:0.3]
# datafile = 'motor.log'

# set multiplot layout 2,2 rowsfirst

# plot for [IDX=3:6] datafile u 1:IDX w lines
# plot for [IDX=7:8] datafile u 1:IDX w lines
# plot for [IDX=9:12] datafile u 1:IDX w lines
