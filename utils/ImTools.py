import numpy as np
import cv2 

'''
# This class implements a collection of image processing operations
'''
class ImTools:
	def __init__(self):
		pass
		
	def bilateralFilter(self, src, d, sigmaColor, sigmaSpace):
		"""Apply bilateral filtering to the source image
		Wraps OpenCV's function: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
		"""
		return cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)

	def contrastEnhancerLAB(self, src):
		"""Enhance image contrast in LAB color space, via adaptive histogram equalization on the L-channel 
		L-channel: representing lightness in the image
		a-channel: representing change in color between red and green
		b-channel: representing change in color between yellow and blue
		CLAHE: Contrast Limited Adaptive Histogram Equalization
		"""
		lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)			# Conversion to LAB color space
		l_channel, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(10,10))	# Applying CLAHE to image L-channel
		cl = clahe.apply(l_channel)
		limg = cv2.merge((cl,a,b))					# Merge the CLAHE enhanced L-channel with the a and b channel
		enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)		#Regress to BGR color space

		return enhanced_img

	def binarizerOSTU(self, src):
		"""Binarize image according to OSTU method, requiring image to be grayscale
		Wraps OpenCV's function: https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
		"""
		#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		_, binarized_img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		return binarized_img
		
	def close(self, src, kernel_size=50):
		"""Perform a closing operation over the image, using OpenCV's morphology functions
		Wraps OpenCV's function: 
		"""
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
		return closed
		
	def naiveSegmenter(self, src):
		
		filtered_img = self.bilateralFilter(src, 15, 75, 75)			#Remove noise through bilateral filtering
		enhanced_img = self.contrastEnhancerLAB(filtered_img)		#Enhance image contrast
		gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
		binarized_img = self.binarizerOSTU(gray_img)				#Binarize Image through OSTU method
		inverted_img = cv2.bitwise_not(binarized_img)				#Invert binarized image
		closed_img = self.close(inverted_img, 3)
		
		return closed_img
		
	def dilate(self, src):
		kernel = np.ones((15, 15), np.uint8)
		dilated = cv2.dilate(src, kernel, iterations = 1)
		return dilated
	
	
		
		
	def maxConvexHull(self, src):
		gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		dilated =  self.dilate(gray)
		closed = self.close(dilated)
		binarized = self.binarizerOSTU(closed)
		
		(cnts, _) = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(cnts, key = cv2.contourArea)
		mask = np.zeros_like(binarized , dtype=np.uint8)
		convexHull = cv2.convexHull(c)
		cv2.fillPoly(mask, pts=[convexHull], color=255)
		
		return mask
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
