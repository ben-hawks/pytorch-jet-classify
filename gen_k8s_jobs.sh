#! /bin/bash

# Expand the template into multiple files, one for each item to be processed.
mkdir -p ./jobs
for p in 32 12 6 4 #8
do
  for i in 0 25 50 75 100
  do
    cat pt-jet-class-job-FT_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/pt-jet-class-job-FT-$i-$p.yaml
    cat pt-jet-class-job-LT_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/pt-jet-class-job-LT-$i-$p.yaml
  done
done