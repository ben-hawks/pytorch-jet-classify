#! /bin/bash

# Expand the template into multiple files, one for each item to be processed.

mkdir -p ./post_jobs
for i in 0 25 50 75 100
do
  cat pt-jet-postprocess-FT_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-FT-$i.yaml
  cat pt-jet-postprocess-LT_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-LT-$i.yaml
  cat pt-jet-postprocess-job-FT_NoBN_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-FT_NoBN-$i.yaml
  cat pt-jet-postprocess-job-FT_NoL1_template.yml | sed "s/\$RAND/$i/" > ./post_jobs/pt-jet-postprocess-job-FT_NoL1-$i.yaml
done