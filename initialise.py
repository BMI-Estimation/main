import random
import colorsys

# load the class label names from disk
CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")
# generate random colors for each class label
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
	# name configuration
	NAME = "coco_inference"
	# set the number of GPUs to use, the number of images per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# number of classes
	NUM_CLASSES = len(CLASS_NAMES)

config = SimpleConfig()

