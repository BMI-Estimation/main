import unittest
import numpy as np
from PIL import Image
import cv2
from binaryMask import mask2binary
from boundingBoxes import midpoint

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

if __name__ == '__main__':
	unittest.main()