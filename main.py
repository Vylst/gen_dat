import numpy as np 
import os
from PIL import Image, ImageEnhance
from skimage import transform as tf
from skimage.util import random_noise
from utils.ImTools import ImTools

class Transformation:
	def __init__(self):
		pass

	def morph_to_background(self, src, fm, background):

		#A reduction should not occur for more than 

		original_back_width = np.shape(background)[0]
		original_back_height = np.shape(background)[1]

		modified = src.resize((int(3*original_back_width/4), int(3*original_back_height/4)))
		modified_mask = fm.resize((int(3*original_back_width/4), int(3*original_back_height/4)))

		return modified, modified_mask
		
	def apply_skew(self, src, fm):
		random_factor = np.random.uniform(0, 1)
		skew = np.random.uniform(-random_factor, random_factor)
		
		width, height = src.size
		xshift = abs(skew) * width
		new_width = width + int(round(xshift))
		src = src.transform((new_width, height), Image.AFFINE, (1, skew, -xshift if skew > 0 else 0, 0, 1, 0))
		fm = fm.transform((new_width, height), Image.AFFINE, (1, skew, -xshift if skew > 0 else 0, 0, 1, 0))

		return src, fm

	def randomize_contrast(self, src):
		enhancer = ImageEnhance.Contrast(src)
		random_factor = np.random.uniform(0, 1) + 0.5
		modified = enhancer.enhance(random_factor)

		return modified

	def stretch_width(self, src, fm):
		original_fence_width = np.shape(src)[0]
		original_fence_height = np.shape(src)[1]
		random_factor = int(np.random.uniform(0, 1) * 100)

		modified = src.resize((original_fence_width + random_factor, original_fence_height))
		modified_mask = fm.resize((original_fence_width + random_factor, original_fence_height))

		return modified, modified_mask

	def random_quadrant_turn(self, src, fm):
		random_factor = 90*np.random.randint(4)
		modified = src.rotate(random_factor)
		modified_mask = fm.rotate(random_factor)
		
		return modified, modified_mask
		
	def tilt(self, src, fm):
		random_factor = np.random.randint(40)-20
		modified = src.rotate(random_factor)
		modified_mask = fm.rotate(random_factor)
  
		return modified, modified_mask
		
	def add_salt_and_pepper(self, image):

		amount = 0.2

		output = np.copy(np.array(image))

		nb_salt = np.ceil(amount * output.size * 0.5)
		coords = [np.random.randint(0, i - 2, int(nb_salt)) for i in output.shape]
		output[coords[0],coords[1],coords[2]] = 1

		nb_pepper = np.ceil(amount* output.size * 0.5)
		coords = [np.random.randint(0, i - 2, int(nb_pepper)) for i in output.shape]
		output[coords[0],coords[1],coords[2]] = 0

		return Image.fromarray(output)

	def to_mask(self, src):
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

		new_img = Image.fromarray(mask)
		return new_img


if __name__ == '__main__':

	tools = ImTools()

	str_cmd = "rm src_fences/.DS_Store"
	os.system(str_cmd)
	str_cmd = "rm src_backgrounds/.DS_Store"
	os.system(str_cmd)

	#Get transformations object
	img_transformer = Transformation()

	paths = ['src_fences_normal', 'src_fences_anomalous']
	for p in paths:
	
		#Load png fence images into memory
		lt_fences = []
		for image_path in os.listdir(p):
			img = Image.open(p + '/' + image_path)
			img = img.convert(mode='RGBA')
			lt_fences.append(img)

		print(len(lt_fences))
		
		label = paths.index(p)

		#Loop through src background images
		bpath = 'src_backgrounds'
		for image_path in os.listdir(bpath):
			img = Image.open(bpath + '/' + image_path)
			img = img.convert(mode='RGBA')

			#Loop through fence images in memory, apply transformations and overlap with background image
			for i in range(len(lt_fences)):

				#Repeat this process 2-3 times so that the same fence can be applied over the same background with different configs
				for k in range(3):

					#Create empty mask for generated image
					mask_path = 'masks'
					mask = Image.fromarray(np.zeros_like(img), mode = 'RGB')

					#Get fence image, adapt to to background size and apply transformations
					fence = lt_fences[i]
					fence_mask = img_transformer.to_mask(fence) #Convert transformed fence image into black and white

					fence, fence_mask = img_transformer.random_quadrant_turn(fence, fence_mask)
					fence, fence_mask = img_transformer.morph_to_background(fence, fence_mask, img)
					fence, fence_mask = img_transformer.apply_skew(fence, fence_mask)
					fence = img_transformer.randomize_contrast(fence)
					fence, fence_mask = img_transformer.stretch_width(fence, fence_mask)
					fence, fence_mask = img_transformer.tilt(fence, fence_mask)
					

					#Randomize location of fence over background (fence should now be approx 50% of background size)
					# (0,0) ----- >
					# |
					# |
					# |
					# v
					if(np.shape(img)[0] - np.shape(fence)[0] > 0):
						ix = np.random.randint(np.shape(img)[0] - np.shape(fence)[0])
					else:
						ix = 0

					if(np.shape(img)[1] - np.shape(fence)[1] > 0):
						iy = np.random.randint(np.shape(img)[1] - np.shape(fence)[1])
					else:
						iy = 0

					#Populate mask for generated image
					mask.paste(fence_mask, (ix,iy), mask = fence_mask)
					mask = mask.point( lambda p: 255 if p > 160 else 0 )

					#Store mask in folder, also flipped vertically and horizontally
					save_path = 'masks'
					n = len([name for name in os.listdir(save_path) if os.path.isfile(save_path + '/' + name)])
					mask.save(save_path + '/' + str(n) + '_' + str(label) + '.png')
					
					hullMask = Image.fromarray(tools.maxConvexHull(np.array(mask)))
					hullMask.save('hulls/' + str(n) + '_' + str(label) + '.png')
					

					mask2 = mask.transpose(Image.FLIP_LEFT_RIGHT)
					mask2.save(save_path + '/' + str(n+1) + '_' + str(label) + '.png')
					
					hullMask = Image.fromarray(tools.maxConvexHull(np.array(mask2)))
					hullMask.save('hulls/' + str(n+1) + '_' + str(label) + '.png')
					
					mask3 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
					mask3.save(save_path + '/' + str(n+2) + '_' + str(label) + '.png')
					
					hullMask = Image.fromarray(tools.maxConvexHull(np.array(mask3)))
					hullMask.save('hulls/' + str(n+2) + '_' + str(label) + '.png')
					
					
					
					
					

					#Paste transformed image over background
					img.paste(fence, (ix,iy), mask = fence)

					#Add noise to generated image
					img = img_transformer.add_salt_and_pepper(img)

					#Store generated image in destination folder, also flipped vertically and horizontally
					save_path = 'generated_images'
					n = len([name for name in os.listdir(save_path) if os.path.isfile(save_path + '/' + name)])
					img.save(save_path + '/' + str(n) + '_' + str(label) + '.png')

					img = img.transpose(Image.FLIP_LEFT_RIGHT)
					img.save(save_path + '/' + str(n+1) + '_' + str(label) + '.png')
					
					img = img.transpose(Image.FLIP_TOP_BOTTOM)
					img.save(save_path + '/' + str(n+2) + '_' + str(label) + '.png')
					
					#Fetch back original background image for next iteration
					img = Image.open(bpath + '/' + image_path)
					img = img.convert(mode='RGBA')
























