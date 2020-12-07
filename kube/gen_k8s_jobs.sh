#! /bin/bash

# Expand the template into multiple files, one for each item to be processed.
mkdir -p ./jobs
mkdir -p ./jobs/FT
mkdir -p ./jobs/LT
mkdir -p ./jobs/BN
mkdir -p ./jobs/NoBN
mkdir -p ./jobs/NoL1
for p in 32 12 6 4 #8
do
  for i in 0 25 50 75 100
  do
    cat pt-jet-class-job-FT_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/FT/pt-jet-class-job-FT-$i-$p.yaml
    cat pt-jet-class-job-LT_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/LT/pt-jet-class-job-LT-$i-$p.yaml
    cat pt-jet-class-job-FT_NoBN_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/NoBN/pt-jet-class-job-LT-$i-$p.yaml
    cat pt-jet-class-job-FT_NoL1_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/NoL1/pt-jet-class-job-LT-$i-$p.yaml
  done
  cat pt-jet-class-job-FT_NoStats_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-FT-NoStats-batnorm-$p.yaml
  cat pt-jet-class-job-LT_NoStats_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-LT-NoStats_batnorm-$p.yaml
  cat pt-jet-class-job-FT_NoAff_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-FT-NoAff-batnorm-$p.yaml
  cat pt-jet-class-job-LT_NoAff_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-LT-NoAff_batnorm-$p.yaml
  cat pt-jet-class-job-FT_NoL1_NoStatsBN_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-FT-NoL1-NoStatsBN-$p.yaml
done