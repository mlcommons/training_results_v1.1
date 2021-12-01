### The Mask R-CNN script operates on COCO, a large-scale object detection, segmentation, and captioning dataset. To download and verify the dataset use following scripts:

```
 cd dataset_scripts
 ./download_dataset.sh
 ./verify_dataset.sh
```

### This should return PASSED. To extract the dataset use:

```
 DATASET_DIR= ./extract_dataset.sh
``` 
### Mask R-CNN uses pre-trained ResNet50 as a backbone. To download and verify the RN50 weights use:

```
 DATASET_DIR= ./download_weights.sh 
```

### Make sure exists and is writable. To speed up loading of coco annotations during training, the annotations can be pickled since unpickling is faster than loading a json.

```
 python pickle_coco_annotations.py --root --ann_file --pickle_output_file
```