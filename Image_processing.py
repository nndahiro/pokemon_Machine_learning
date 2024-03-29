'''
This is where all the image processing functions are defined
'''
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from tqdm import tqdm
import skimage.transform as transform
from skimage import filters
from skimage.measure import label
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.color import rgb2grey
from scipy import ndimage as ndi


def array_image(fptr, normalize = True, verbose=True,transparent = False, hsv=False):
	'''Converts an RGB-pixel image to an array of type
	defined the array is accessible as:
	array[height][width][color:red,gree,or blue] OR:
	array[rows][columns][colorsRGB]. You can also
	normalize the units of the pixels using this file.
	**Parameters**:
		fptr: *str*
			This is the path of the image you want to
			convert into an array.
		normalize: *bool*
			If set to True, the image pixels will be
			normalized.
		verbose:*bool*
			If set to true it will print out information
			about the original and final image.
		transparent:*bool*
			If one wants to plot images in RGBA then your
			array will be converted to RGBA.
		hsv: *bool*
			If you want to output the array in HSV (Hue,
			Saturation, Value) then the function will
			output an array in HSV. However, this can only
			be done if the array is to be normalized as well.
			RGB->HSV works when normalized.
	**Returns**
		array_i: *numpy.array*, *float* or *int*
			The image that has been turned to an array.

	'''
	if transparent:
		image = Image.open(fptr).convert("RGBA") #Image object
		width, height = image.size

	else:
		image = Image.open(fptr).convert("RGB") #Image object
		width, height = image.size
	
		
	array_i = np.asarray(image)
	
	array_i = array_i.astype(np.float32) # Ensures image is not
	# read-only so it's normalizeable
	if normalize:
		array_i /= 255

	if hsv:
		if normalize is False:
			print('Can Only convert to HSV if normalize is True')

	if hsv and normalize:#To turn to HSV, image must be normalized 
		array_i = rgb_hsv(array_i)
	if verbose:
		print(f"Image path is {fptr}")
		print(f"IMAGE size : {width} x {height}")	
		print((array_i[0][0][0]))
		print(f"Array shape: {array_i.shape}")
	return array_i

def pixel_reduce(image_array, mask):
	'''
	An image augmentation that blurs the image
	by a certain factor defined by mask.
	**Parameters**
		image_array: *numpy.array*,*float* or *int*
			The array containing an image to be
			augmented.
		mask: *int* or *float*
			The factor by which the image will be
			blurred. Will round up or down to closest
			integer.
	**Returns**
		new_image: *numpy.array*
			The augmented image

	'''

	new_image = transform.resize(image_array, (image_array.shape[0] // int(mask), \
                                                image_array.shape[1] // int(mask)), \
                                    anti_aliasing=True)
	return new_image
def image_array_resize(image_array,size):
	'''
	Resizes image to any size needed. It applies an image
	augmentation function with anti-alising where not much
	contrast is lost. So the different features of a
	pokemon image remain the same colors. The shape of the
	array should be	(n,m,3) for colored images and (n,m)
	for black and white images.
	**Parameters**
		image_array: *np.array*,*float* or *int*
			This is the image data in an array. the
			image pixels will usually be normalized
			but it works for both.
		size: *tuple* *int*
			This is the target size of your image.
	**Returns**
		new_image: **numpy.array*,*float* or *int*
			The new, resized image.
	'''
	new_image = transform.resize(image_array,size)
	return new_image

def centralize_image(image_array, show=False):
	'''
	Takes image, finds object in image by: 1) finding
	edges based on intensity gradient in picture, 2)
	finds coordinates of object and 3) returns "centralized"
	image. The different stages of the processing can be
	shown by setting show=True.
	**Parameters**
		image_array: *numpy.array* *float* or *int*
				The image information in array form with
				any size.
		show: *Bool*
				Whether or not to plot the graphs of the
				different processing steps.
	**Returns**
		new_image_r: *numpy.array* *float* or *int*
				The centralized image in an array form.
	'''
	gray_image = rgb2grey(image_array)
	if show:
		plt.subplot(3,3,1)
		plt.title('Grayscale')
		plt.imshow(gray_image, cmap = plt.cm.gray)
	edges = filters.sobel(gray_image) #Get edges, faster to do it when gray
	if show:
		plt.title('Sobel Filtering')
		plt.subplot(3,3,2)
		plt.imshow(edges, cmap = plt.cm.gray)

	binary_image = filters.threshold_local(edges, block_size=3) #Turn to binary for filling holes
	if show:
		plt.title('Binary')
		plt.subplot(3,3,3)
		plt.imshow(binary_image, cmap = plt.cm.gray)

	holes_filled = ndi.binary_fill_holes(binary_image) # fill holes
	if show:
		plt.title('Remove empty space')
		plt.subplot(3,3,4)
		plt.imshow(holes_filled, cmap = plt.cm.gray)

	selected_image = label(holes_filled) # Most empty space is removed
	if show:
		plt.title('Contour found')
		plt.subplot(3,3,5)
		plt.imshow(selected_image, cmap = plt.cm.gray)
	coordinate_slices = ndi.measurements.find_objects(selected_image)[0]
	# Returns Slice of min x,y and max x,y
	# can be directly inputed into image_array
	
	new_image = image_array[coordinate_slices]
	if show:
		plt.title('Cropped image')
		plt.subplot(3,3,6)
		plt.imshow(new_image)
		plt.show()

	new_image_r = image_array_resize(new_image, (100,100,3))
	return new_image_r

def rgb_hsv(normalized_array):
	'''A function that runs the rgb_to_hsv function 
	from the matplotlib module. It only takes in
	normalized RGB pixels.
	**Parameters**
		normalized_array: *numpy.array*, *float*
			This is an array of pixels in RGB that
			has been normalized toa range [0,1]
	**Returns**
		hsv_array: *numpy.array*
			The new hsv image.
	'''
	hsv_array = pltcolors.rgb_to_hsv(normalized_array)
	return hsv_array

if __name__ == "__main__":

	print("Image Processing in place...")