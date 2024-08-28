import cv2
import numpy as np

img = cv2.imread('10.png')

kernel = np.ones((1, 1), np.uint8)
dilated = cv2.dilate(img, kernel, iterations = 1)


img = cv2.cvtColor(dilated, cv2.COLOR_RGB2RGBA)


for i in range(np.shape(img)[0]):
	for j in range(np.shape(img)[1]):
		if(img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0):
			img[i,j,3] = 0


cv2.imwrite('t.png', img)




