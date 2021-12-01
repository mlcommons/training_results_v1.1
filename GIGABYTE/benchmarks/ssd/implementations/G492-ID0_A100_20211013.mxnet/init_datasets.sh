### Steps to download data & model
### The SSD script operates on COCO, a large-scale object detection, segmentation, and captioning dataset. To download the dataset run the following cmds:

```
DATASET_DIR=
COCO_DIR=$DATASET_DIR/coco2017

mkdir -p $COCO_DIR

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d $COCO_DIR
unzip val2017.zip -d $COCO_DIR
unzip annotations_trainval2017.zip -d $COCO_DIR

rm train2017.zip val2017.zip annotations_trainval2017.zip
```

### Additionally, create these SSD-specific annotation files:
````
python3 prepare-json.py --keep-keys $COCO_DIR/annotations/instances_val2017.json $COCO_DIR/annotations/bbox_only_instances_val2017.json 
python3 prepare-json.py --keep-keys $COCO_DIR/annotations/instances_train2017.json $COCO_DIR/annotations/bbox_only_instances_train2017.json
```
### (Or Next download and convert the required reference ResNet-34 pretrained backbone weights (which, in the reference, come from TorchVision) into a format readable by non-pytorch frameworks. From the directory containing this README run:)
```
./scripts/get_resnet34_backbone.sh
```