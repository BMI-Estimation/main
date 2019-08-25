import cv2
from initialise import model, CLASS_NAMES, color
from mrcnn import visualize
from personMetrics import extractMaskFromROI

def findPersonInPhoto(image, show, showMask):
	# convert to rgb image for model
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# perform forward pass of the network
	print("[INFO] making predictions with Mask R-CNN...")
	r = model.detect([image], verbose=1)[0]
	# loop over of the detected object's bounding boxes and masks
	for i in range(0, r["rois"].shape[0]):
	 	# extract the class ID and mask
		classID = r["class_ids"][i]
		# ignore all non-people objects
		if CLASS_NAMES[classID] != 'person':
			continue

		clone = image.copy()
		mask = r["masks"][:, :, i]
		# visualize the pixel-wise mask of the object
		image = visualize.apply_mask(image, mask, color, alpha=0.5)
		# convert the image to BGR for OpenCV use
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		(startY, startX, endY, endX) = r["rois"][i]
			
		# extract the ROI of the image, use foreground extraction for mask
		roi = clone[startY:endY, startX:endX]
		roiMask = extractMaskFromROI(roi)
		# extract the mask produced by CNN
		visMask = (mask * 255).astype("uint8")
		visMask = visMask[startY:endY, startX:endX]
		# extract overlapping regions of both masks to minimalise mask errors.
		finalMask = cv2.bitwise_and(roiMask, visMask)

		if show or showMask:
			if show:
				cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
				cv2.imshow("ROI", roi)
				cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
				cv2.imshow("Output", image)
				cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
				cv2.imshow("Mask", visMask)
				cv2.namedWindow('roi mask', cv2.WINDOW_NORMAL)
				cv2.imshow('roi mask', roiMask)
			if showMask:
				cv2.namedWindow('final mask', cv2.WINDOW_NORMAL)
				cv2.imshow('final mask', finalMask)
			cv2.waitKey(0)
		break
	return finalMask
