#This script serves to concatenate an image of a background with a fence and its correpsonding binary mask, horizontally side by side
import numpy as np
import os
from PIL import Image

if not os.path.exists('hull_dataset'):
	os.makedirs('hull_dataset')

path = 'generated_images'
for image_name in os.listdir(path):
	
	#Read generated image
	img = Image.open(path + '/' + image_name)
	img = img.convert(mode='RGB')

	#Read mask image
	mask = Image.open('hulls/' + image_name)
	mask = mask.convert('RGB')
	
	
	#Concat the two
	dummy = np.zeros((np.shape(img)[0], 2*np.shape(img)[1], np.shape(img)[2]))
	dummy[:,0:np.shape(img)[1],:] = img
	dummy[:,np.shape(img)[1]:,:] = mask
	
	dummy = dummy.astype(np.uint8)
	dummy = Image.fromarray(dummy)
	
	dummy.save('hull_dataset/' + image_name)
	
	











































