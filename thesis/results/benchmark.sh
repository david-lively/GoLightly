
for((w=128; w <= 8192; w+=128))
 do 
 echo "width " $w;
 echo "height " $((w / 2));

 meep dw=$w dh=$((w/2)) benchmark.ctl
 done;
 
