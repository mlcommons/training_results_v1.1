###　Start the container interactively, mounting the directory you want to store the expieriment data as /workspace/bert_data

```
docker run -it --runtime=nvidia --ipc=host (...) -v /data/mlperf/bert:/workspace/bert_data mlperf-nvidia:language_model
```

### Within the container, run

```
cd /workspace/bert
./input_preprocessing/prepare_data.sh --outputdir /workspace/bert_data
```

### This script will download the required data and model files from MLCommons members Google Drive location creating the following foldes structure

```
/workspace/bert_data/
                     |_ download
                         |_results4 # 500 chunks with text data 
                         |_bert_reference_results_text_md5.txt # md5 checksums for text chunks
                     |_ phase1 # checkpoint to start from (both tf1 and pytorch converted)
                     |_hdf5 
                             |_ eval # evaluation chunks in binary hdf5 format fixed length (not used in training, can delete after data preparation) 
                             |_ eval_varlength # evaluation chunks in binary hdf5 format variable length *used for training*
                             |_ training # 500 chunks in binary hdf5 format 
                             |_ training_4320 # 
                                 |_ hdf5_4320_shards_uncompressed # sharded data in hdf5 format fixed length (not used in training, can delete after data preparation)
                                 |_ hdf5_4320_shards_varlength # sharded data in hdf5 format variable length *used for training*
```

### The resulting HDF5 files store the training/evaluation dataset as a variable-length types (https://docs.h5py.org/en/stable/special.html). Note that these do not have a direct Numpy equivalents and require "by-sample" processing approach. The advantage is significant storage requirements reduction. The prepare_data.sh script does the following:

    1. downloads raw data from GoogleDrive
    2. converts the training data to hdf5 format (each of 500 data chunks)
    3. splits the data into appropriate number of shards (for large scale training we recommend using 4320 shards - the default)
    4. 'compresses' the shards converting fixed-length hdf5 to variable-length hdf5 format
    5. applies the same procedure to evaluation data
    6. converts the seed checkpoint from tensorflow 1 to pytorch format

### To verify correctness of resulting data files one may compute checksums for each of shards (using hdf5_md5.py script) and compere it with checksums in 4320_shards_varlength.chk or 2048_shards+varlength.chk files. Example of how to compute the checksums

### Generate checksums to verify correctness of the process possibly paralellized with e.g. xargs and then sorted
```
for i in `seq -w 0000 04319`; do 
 python input_preprocessing/hdf5_md5.py \
 --input_hdf5 path/to/varlength/shards/part_${i}_of_04320.hdf5 
done | tee 4320_shards_varlength.chk
```