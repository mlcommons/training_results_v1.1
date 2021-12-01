### Steps to download and verify data
### Build a data preprocessing container using Dockerfile_pyt, and run it

```
sudo chown -R $USER:$USER scripts
cd scripts
docker build -t preprocessing -f Dockerfile_pyt .
docker run --ipc=host -it --rm --runtime=nvidia -v DATASET_DIR:/data preprocessing:latest 
```
### Where DATASET_DIR is the target directory used to store the dataset after preprocessing.

### Download and preprocess the data inside the container
```
bash download_dataset.sh 
```

### After this, the preprocessed data in .npy format will be available in the taget directory (outside the container)

### exit the container