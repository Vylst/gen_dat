import numpy as np
import cv2
from PIL import Image
from skimage import transform as tf
from utils.ImTools import ImTools

def to_mask(src):
	src = np.asarray(src)
	mask = np.zeros_like(src)
	for i in range(np.shape(src)[0]):
		for j in range(np.shape(src)[1]):
			if(src[i,j,3] != 0):
				mask[i,j,0] = 255
				mask[i,j,1] = 255
				mask[i,j,2] = 255
				mask[i,j,3] = 255
			else:
				mask[i,j,0] = 0
				mask[i,j,1] = 0
				mask[i,j,2] = 0
				mask[i,j,3] = 255

	return mask

def remove_background(src):
	arg = 230
	src = np.asarray(src)
	mask = np.zeros_like(src)
	for i in range(np.shape(src)[0]):
		for j in range(np.shape(src)[1]):
			if(src[i,j,0] > arg and src[i,j,1] > arg and src[i,j,2] > arg):
				mask[i,j,0] = 0
				mask[i,j,1] = 0
				mask[i,j,2] = 0
				mask[i,j,3] = 0
			else:
				mask[i,j,0] = src[i,j,0]
				mask[i,j,1] = src[i,j,1]
				mask[i,j,2] = src[i,j,2]
				mask[i,j,3] = src[i,j,3]

	return mask


def skew_image(src):
	random_factor = np.random.uniform(0, 1) * 2
	skew = np.random.uniform(-random_factor, random_factor)
	
	width, height = src.size
	xshift = abs(skew) * width
	new_width = width + int(round(xshift))
	src = src.transform((new_width, height), Image.AFFINE, (1, skew, -xshift if skew > 0 else 0, 0, 1, 0))

	return src
	
def add_salt_and_pepper(image):

	amount = 0.2

	output = np.copy(np.array(image))

	print(output.shape)

	nb_salt = np.ceil(amount * output.size * 0.5)
	coords = [np.random.randint(0, i - 2, int(nb_salt)) for i in output.shape]
	print(coords[0])
	output[coords[0],coords[1],coords[2]] = 1

	nb_pepper = np.ceil(amount* output.size * 0.5)
	coords = [np.random.randint(0, i - 2, int(nb_pepper)) for i in output.shape]
	output[coords[0],coords[1],coords[2]] = 0

	return Image.fromarray(output)



tools = ImTools()
img = cv2.imread('mask.png')
hullMask = tools.maxConvexHull(img)


#img = img.convert(mode='RGBA')
#new_img = to_mask(img)
#new_img = remove_background(img)
#new_img = Image.fromarray(new_img)
#new_img.save("teste.png")


cv2.imshow('test2', hullMask)
cv2.waitKey(0)
cv2.destroyAllWindows()







