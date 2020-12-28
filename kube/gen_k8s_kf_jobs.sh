#! /bin/bash

# Expand the template into multiple files, one for each item to be processed.
mkdir -p ./jobs
mkdir -p ./jobs/KF
prec=(32 12 6 4) #8
kfolds=(1 2 3 4)
rand=(0 50 75 90)
for k in "${kfolds[@]}"
do
  for p in "${prec[@]}"
  do
    for i in "${rand[@]}"
    do
      cat pt-jet-class-job-kf-FT_template.yml | sed "s/\$K/$k/" | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/KF/pt-jet-class-job-kf-FT-K"$k"-"$i"-"$p".yaml
      cat pt-jet-class-job-kf-LT_template.yml | sed "s/\$K/$k/" | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/KF/pt-jet-class-job-kf-LT-K"$k"-"$i"-"$p".yaml
      cat pt-jet-class-job-kf-FT_NoBN_template.yml | sed "s/\$K/$k/" | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/KF/pt-jet-class-job-kf-FT_NoBN-K"$k"-"$i"-"$p".yaml
      cat pt-jet-class-job-kf-FT_NoL1_template.yml | sed "s/\$K/$k/" | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/KF/pt-jet-class-job-kf-FT_NoL1-K"$k"-"$i"-"$p".yaml
    done
  done
done

# split jobs into 52 job batches (k8s overlords said keep concurrent jobs to ~50)
dir_size=52
dir_name="kfold_batch_"
cd ./jobs/KF
n=$((`find . -maxdepth 1 -type f | wc -l`/$dir_size+1))
for i in `seq 1 $n`;
do
    mkdir -p "$dir_name$i";
    find . -maxdepth 1 -type f | head -n $dir_size | xargs -i mv "{}" "$dir_name$i"
done