# Run me on a kube pod!!!!!!!

# In case of any job-reruns, the model files etc. still exist from the incomplete/previous jobs
# this steps through every model directory and moves all but the most recent set of 12 models
# to a folder called old/ so that when running postprocessing, things still work fine.

prec=(32 12 6 4) #8
kfolds=(1 2 3 4)
rand=(0 50 75 90)
for k in "${kfolds[@]}"
do
  for i in "${rand[@]}"
  do
    for p in "${prec[@]}"
    do
		(cd /ptjetclassvol/train_output/kfold/val_"$k"/FT_"$i"/models/"p"b && mkdir ./old/ &&
	ls -tp | grep -v '/$' | tail -n +13 | xargs -I {} mv -- {} ./old/)
		(cd /ptjetclassvol/train_output/kfold/val_"$k"/FT_"$i"_NoBN/models/"p"b && mkdir ./old/ &&
	ls -tp | grep -v '/$' | tail -n +13 | xargs -I {} mv -- {} ./old/)
		(cd /ptjetclassvol/train_output/kfold/val_"$k"/FT_"$i"_NoL1/models/"p"b && mkdir ./old/ &&
	ls -tp | grep -v '/$' | tail -n +13 | xargs -I {} mv -- {} ./old/)
		(cd /ptjetclassvol/train_output/kfold/val_"$k"/LT_"$i"/models/"p"b && mkdir ./old/ &&
	ls -tp | grep -v '/$' | tail -n +13 | xargs -I {} mv -- {} ./old/)
	done
  done
done
