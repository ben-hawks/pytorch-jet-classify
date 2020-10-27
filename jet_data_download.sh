# Download just the test dataset, create train_data if it doesn't exist, extract all files to dir train_data/test/
curl -o jets_test.tar.gz https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz && mkdir -p train_data && tar -xzvf jets_test.tar.gz --transform "s/val/test/" -C train_data/
curl -o jets_train.tar.gz https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz && mkdir -p train_data && tar -xzvf jets_train.tar.gz -C train_data/
rm jets_test.tar.gz && rm jets_train.tar.gz
