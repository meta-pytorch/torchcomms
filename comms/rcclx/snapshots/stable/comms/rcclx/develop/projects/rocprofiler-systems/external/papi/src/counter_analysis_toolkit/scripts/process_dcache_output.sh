#!/bin/bash

if (($# != 1 )); then
    echo "Usage: process_dcache_output.sh datafile"
fi

ifile=$1

awk 'BEGIN{
  first=1;
}{
  sum=0;
  min=-1;
  max=-1;
  # Process all lines that are not comments.
  if($1!="#"){
    first=0;
    # Find the min, max and sum of measurements across threads.
    for(i=2; i<=NF; i++){
      sum+=$i;
      if($i<min || min<0)
        min=$i;
      if($i>max)
        max=$i;
    }
    # Print only the size and three statistics
    print $1" "min" "sum/(NF-1)" "max;
  }else{
    # Add a space between data blocks to facilitate gnuplot.
    if(!first){
        printf("\n");
    }
    printf("%s\n",$0);
  }
}' $ifile > ${ifile}.stat
