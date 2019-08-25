from mrcnn.config import Config
from mrcnn import model as modellib

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

print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
model.load_weights("mask_rcnn_coco.h5", by_name=True)
