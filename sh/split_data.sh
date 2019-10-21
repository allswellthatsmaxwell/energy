#!/bin/bash
### Splits a data file into training and validation set.
### Takes one argument - the size of the training set.
### The method is kinda bad - just takes the first N rows
### as train, the rest as validation - but if the file
### is ordered by site, or time, or something, we'll be
### out-of-sample over a lot of our validation set.

## this one must exist
total_file=../intermediate/combined_data_16000000.csv
## these are (over)written to
val_file=../intermediate/combined_val.csv
train_file=../intermediate/combined_train.csv
train_size=$1
total_size=`wc -l $total_file | cut -f1 -d ' '`
total_size=`echo $total_size - 1 | bc` ## uncount header
val_size=`echo "$total_size - $train_size" | bc`
header=`head -1 $total_file`
head -"$train_size" "$total_file" >> "$train_file"

echo $header > $val_file
tail -"$val_size" "$total_file" >> "$val_file"
