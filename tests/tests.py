import unittest
import numpy as np
from PIL import Image
import cv2
from binaryMask import mask2binary
from boundingBoxes import midpoint
from personMetrics import personArea, maskThickness
from trainingFunctions import overallscore, baseline_model

class TestBinaryMaskMethods(unittest.TestCase):
	def test_mask2binary(self):
		w, h = 512, 512
		data = np.zeros((h, w, 3), dtype=np.uint8)
		data[100:200, 100:200] = [100, 100, 100]
		testImg = Image.fromarray(data, 'RGB')
		testImg = np.array(testImg)
		testImg = testImg[:, :, ::-1].copy()

		data = np.zeros((h,w), dtype=np.uint8)
		data[100:200, 100:200] = 255
		resultImg = Image.fromarray(data, 'L')
		resultImg = np.array(resultImg)

		self.assertTrue(np.array_equal(mask2binary(testImg), resultImg), "mask2binary method successfully converts RGB image mask to Binary image.")

	# def test_isupper(self):
	# 	self.assertTrue('FOO'.isupper())
	# 	self.assertFalse('Foo'.isupper())

	# def test_split(self):
	# 	s = 'hello world'
	# 	self.assertEqual(s.split(), ['hello', 'world'])
	# 	# check that s.split fails when the separator is not a string
	# 	with self.assertRaises(TypeError):
	# 		s.split(2)

class TestBoundingBoxMethods(unittest.TestCase):
	def test_midpoint(self):
		self.assertEqual(midpoint([0,0] , [0, 10]), (0,5))
		self.assertEqual(midpoint([0,0] , [10, 0]), (5,0))
		self.assertEqual(midpoint([0,0] , [10, 10]), (5,5))
		self.assertEqual(midpoint([0,0] , [0, 5]), (0, 2.5))
		self.assertEqual(midpoint([0,0] , [5, 0]), (2.5, 0))

class TestPersonMetrics(unittest.TestCase):
  def test_personArea(self):
    w, h = 512, 512
    data = np.zeros((h,w), dtype=np.uint8)
    # 100 by 100 pixel square
    data[100:200, 100:200] = 255
    resultImg = Image.fromarray(data, 'L')
    resultImg = np.array(resultImg)
    # at 1m per pixel length, this should translate to 10000m^2
    self.assertEqual(personArea(resultImg, 1), 10000)

class TestTrainingFunctions(unittest.TestCase):
  def test_overallscore(self):
    # when MAE > 4, no extra weighting is granted
    self.assertEqual(overallscore(5, 10), 0.5)
    self.assertEqual(overallscore(4, 8), 0.5)
    # every 1 MAE under 4 relates to an extra point of weighting
    self.assertEqual(overallscore(3, 6), 1.5)
    self.assertEqual(overallscore(2,4), 2.5)
    self.assertEqual(overallscore(1,2), 3.5)
    self.assertEqual(overallscore(1,1), 4)
    self.assertEqual(overallscore(0.5,0.5), 4.5)

  def test_maskthickness(self):
    w, h = 512, 512
    data = np.zeros((h,w), dtype=np.uint8)
    # 100 by 100 pixel square
    data[100:200, 100:200] = 255
    resultImg = Image.fromarray(data, 'L')
    resultImg = np.array(resultImg)
    output = maskThickness(resultImg, 1)
    # area
    self.assertEqual(output[0], 10000)
    # height
    self.assertEqual(output[1], 100)
    # 4 max slices of 100 width square
    self.assertEqual(output[2], 100)
    self.assertEqual(output[3], 100)
    self.assertEqual(output[4], 100)
    self.assertEqual(output[5], 100)

  def test_baselineModel(self):
    build_fn = baseline_model(1,[3,2])
    test_model = build_fn()
    NetworkArc = [row.units for row in test_model.model.layers]
    self.assertEqual(NetworkArc, [3,2,1])

if __name__ == '__main__':
	unittest.main()