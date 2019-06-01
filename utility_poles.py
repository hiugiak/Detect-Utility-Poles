"""
Mask R-CNN
Train on Utility Poles dataset.

Copyright (c) 2019 Zhou Xiaojie.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla 

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import json
import numpy as np
import skimage.draw
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import matplotlib.pyplot as plt

# Import Mask RCNN
sys.path.append("Mask_RCNN")  # To find local version of the library
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from mrcnn.visualize import display_instances

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save results of detection, if not provided
# through the command line argument --output
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "result")

############################################################
#  Configurations
############################################################


class UtilityPoleConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "poles"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + pole

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1280


############################################################
#  Dataset
############################################################

class UtilityPoleDataset(utils.Dataset):
    def load_pole(self, dataset_dir, subset, return_coco=False):
        """Load a subset of the Utility Pole dataset.
        dataset_dir: The root directory of the Utility Pole dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]
        coco = COCO("{}/{}/annotations.json".format(dataset_dir, subset))
        image_dir = "{}/{}/".format(dataset_dir, subset)

        # All classes
        class_ids = sorted(coco.getCatIds())

        # All images
        image_ids = list(coco.imgs.keys())

        # Add classes. We have only one class to add.
        self.add_class("pole", 0, "pole")

        # Add images
        for i in image_ids:
            self.add_image(
                "pole", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a utility pole image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pole":
            return super(UtilityPoleDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "pole.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones(
                            [image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(UtilityPoleDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pole":
            return info["path"]
        else:
            super(UtilityPoleDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def detect(model, image_path, output_dir=DEFAULT_OUTPUT_DIR):
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    results = model.detect([image], verbose=1)[0]

    output_json = os.path.join(output_dir, "results.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save result in json file
    with open(output_json, 'w') as f:
        results_json = json.dumps(results)
        f.write(results_json)
    
    # Load val dataset to get class names
    dataset_val = UtilityPoleDataset()
    dataset_val.load_pole(args.dataset, "val")

    _, ax = plt.subplots(1, figsize=(16, 16))
    display_instances(image, results['boxes'], results['masks'],
                      results['class_ids'], dataset_val.class_names, ax=ax)
    # Save image with masks applied
    plt.savefig(os.path.join(output_dir, 'masks_applied.jpg'))


def train(model, epoch, layers, learning_rate):
    """Train the model."""
    # Training dataset.
    dataset_train = UtilityPoleDataset()
    dataset_train.load_pole(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = UtilityPoleDataset()
    dataset_val.load_pole(args.dataset, "val")
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=epoch,
                layers=layers)


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "pole"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    :return: cocoEval: COCOEval, cocoResults: [N, (image_id, category_id, bbox, score, segmentation)]
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

    return cocoEval, results

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse
    import re

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on utility poles dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('-d', '--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the utility pole dataset')
    parser.add_argument('-w', '--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('-e', '--epoch', required=False,
                        type=int, metavar='N',
                        help='Epoches to train and val'),
    parser.add_argument('-l', '--layers', required=False,
                        default='all',
                        help="Select which layers to train, 'heads', '3+', '4+', '5+' or 'all' (default)")
    parser.add_argument('-L', '--learningrate', required=False,
                        default=UtilityPoleConfig.LEARNING_RATE,
                        type=float,
                        help="Learning rate for training (default=0.001)"),
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('-i', '--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect utility pole on')
    parser.add_argument('-o', '--output', required=False,
                        default=DEFAULT_OUTPUT_DIR,
                        metavar="path or output directory",
                        help='Directory the result of detection saved in')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
        assert args.epoch, "Argument --epoch is required for training"
    elif args.command == "detect":
        assert args.image, "Provide --image to detect utility pole"

    assert re.match(r"^(([3-5]\+)|(heads)|(all))$", args.layers), \
        "Argument --layers could only be 'heads', '3+', '4+', '5+' or 'all'"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = UtilityPoleConfig()
    else:
        class InferenceConfig(UtilityPoleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            # DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or detect
    if args.command == "train":
        train(model, args.epoch, args.layers, args.learningrate)
    elif args.command == "detect":
        # TODO:Detect on a image
        detect(model, args.image, args.output)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
