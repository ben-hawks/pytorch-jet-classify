# pytorch-jet-classify

## Requirements:
Python 3.7

```
pip install -r requirements.txt
```

## Training:

```
python3 train.py -i <input H5 File/Directory of Files to train on> -o <output directory> -t <H5 File/Directory of files to use as test set> -e <num epoch to train for> -c <config to use>
```

Upon training completion, graphs for the ROC AUC vs Epoch, Loss vs Epoch, Precision vs Epoch, ROC for each tagger, and Confusion Matrix are saved to the output directory, along with a .pt saved model file. 

The Float/Unquantized 3 Layer model is `models.three_layer_model()`, with the Quantized/Brevitas 3 Layer Model is `models.three_layer_model_bv()`. Either can be chosen to train by setting `current_model` to one of the two. 

At the moment, the precision of `models.three_layer_model_bv()` is set by `self.weight_precision` within the class in `models.py`, though this is likely to change in the future

## PRP Nautilus Kubernetes cluster (https://nautilus.optiputer.net/) instructions:

First create the persistent volume claim (PVC):
```
kubectl create -f pt-jet-class-vol.yml
```
This is used to store the data and model outputs so they persist after deleting pods and jobs.

To do interactive work:
```
# create the pod
kubectl create -f pt-jet-class-pod.yml
# login to the pod
kubectl exec -it pt-jet-class-pod bash
```

In particular, you can populate the PVC with the data:
```
cd /ptjetclassvol/
mkdir data
wget https://raw.githubusercontent.com/ben-hawks/pytorch-jet-classify/master/jet_data_download.sh
source jet_data_download.sh
```

To check on running pods:
```
kubectl	get pods
kubectl	describe pods pt-jet-class-pod
```

To delete the pod:
```
kubectl delete pods pt-jet-class-pod
```
It also auto-deletes after 6 hours.

To launch a job:
```
kubectl create -f pt-jet-class-job.yml
```

To check on running jobs:
```
kubectl get jobs
```

You can also get the logs of the running jobs by getting the pod name first through

```
# get job's pod name
kubectl get pods
kubectl describe jobs pt-jet-class-job
# with pod's name, get logs
kubectl logs pt-jet-class-job-baseline-<random-string>
```





