#! /bin/bash

# Expand the template into multiple files, one for each item to be processed.
mkdir -p ./jobs
mkdir -p ./jobs/FT
mkdir -p ./jobs/LT
mkdir -p ./jobs/BN
mkdir -p ./jobs/NoBN
mkdir -p ./jobs/NoL1
mkdir -p ./jobs/BO
prec=(32 12 6 4) # 8
bo_best=("46,10,17" "61,7,23" "32,11,21" "56,19,31") #Probably a 'nicer' way of loading these, its fine for this scope
rand=(0 50 75 90) # 25 100
for p in "${prec[@]}"
do
  for i in "${rand[@]}"
  do
    cat pt-jet-class-job-FT_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/FT/pt-jet-class-job-FT-"$i"-"$p".yaml
    cat pt-jet-class-job-LT_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/LT/pt-jet-class-job-LT-"$i"-"$p".yaml
    cat pt-jet-class-job-FT_NoBN_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/NoBN/pt-jet-class-job-FT_NoBN-"$i"-"$p".yaml
    cat pt-jet-class-job-FT_NoL1_template.yml | sed "s/\$RAND/$i/" | sed "s/\$PREC/$p/" > ./jobs/NoL1/pt-jet-class-job-FT_NoL1-"$i"-"$p".yaml
  done
  cat pt-jet-class-job-FT_NoStats_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-FT-NoStats-batnorm-"$p".yaml
  cat pt-jet-class-job-LT_NoStats_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-LT-NoStats_batnorm-"$p".yaml
  cat pt-jet-class-job-FT_NoAff_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-FT-NoAff-batnorm-"$p".yaml
  cat pt-jet-class-job-LT_NoAff_batnorm_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-LT-NoAff_batnorm-"$p".yaml
  cat pt-jet-class-job-FT_NoL1_NoStatsBN_template.yml | sed "s/\$PREC/$p/" > ./jobs/BN/pt-jet-class-job-FT-NoL1-NoStatsBN-"$p".yaml
  cat pt-jet-bo-job_template.yml | sed "s/\$PREC/$p/" > ./jobs/BO/pt-jet-bo-job-"$p".yaml
done

for r in "${rand[@]}"
  do
    for ((i=0;i<${#prec[@]};++i)); do
        p=${prec[i]}
        b=${bo_best[i]}

        cat pt-jet-class-job-BO-best-train_template.yml | sed "s/\$RAND/$r/" | sed "s/\$BEST/$b/" | sed "s/\$PREC/$p/" > ./jobs/BO/pt-jet-class-job-BO-best-train-"$p"-"$r".yaml
    done
done