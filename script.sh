#!/bin/bash

for ((i=1;i<=10;i++)); 
do 
   echo $i
   python r0123456.py >> multiple_tries4.csv
done
