
Vietnamese Traffic Signs Detection - v7 2022-11-30 9:53pm
==============================

This dataset was exported via roboflow.com on June 19, 2024 at 5:27 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2417 images.
Traffic-Signs are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-contrast via contrast stretching

The following augmentation was applied to create 3 versions of each source image:
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 5 pixels
* Salt and pepper noise was applied to 2 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Random brigthness adjustment of between -20 and +20 percent


