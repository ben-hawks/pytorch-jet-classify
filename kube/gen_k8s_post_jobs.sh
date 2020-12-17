#! /bin/bash

# Expand the template into multiple files, one for each item to be processed.

mkdir -p ./post_jobs
mkdir -p ./post_jobs/ts/
for i in 0 25 50 75 90 100
do
  cat pt-jet-postprocess-job-FT_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-FT-$i.yaml
  cat pt-jet-postprocess-job-LT_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-LT-$i.yaml
  cat pt-jet-postprocess-job-FT_NoBN_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-FT_NoBN-$i.yaml
  cat pt-jet-postprocess-job-FT_NoL1_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-FT_NoL1-$i.yaml

  cat pt-jet-postprocess-job-FT-trainset_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/ts/pt-jet-postprocess-job-FT-trainset-$i.yaml
  cat pt-jet-postprocess-job-FT_NoBN-trainset_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/ts/pt-jet-postprocess-job-FT_NoBN-trainset-$i.yaml
  cat pt-jet-postprocess-job-FT_NoL1-trainset_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/ts/pt-jet-postprocess-job-FT_NoL1-trainset-$i.yaml
done