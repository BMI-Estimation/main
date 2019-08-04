import colorsys
from mrcnn.config import Config

# load the class label names from disk
CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")
color = (0, 255, 0)

class SimpleConfig(Config):
	# name configuration
	NAME = "coco_inference"
	# set the number of GPUs to use, the number of images per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# number of classes
	NUM_CLASSES = len(CLASS_NAMES)

config = SimpleConfig()

