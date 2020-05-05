'''
EN.640.635 Software Carpentry
Lab 2 - PIL Image Blurring and Luminance

In this lab assignment, we want to write two functions that can manipulate
images: one to blur the image and one to set the luminance (or brightness) of
the image.
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


def blur(fptr, mask):
#fptr is your file name. Mask is the size of the blur you wish to create around each given pixel.
#It has to be an odd number since you are creating a square around 1 pixel.
    
    original_img = Image.open(fptr).convert("RGB")
    img = Image.open(fptr).convert("RGB")
    mask_range = int((mask-1)/2)

    width, height = img.size
    for x in range(width):
        for y in range(height):
            red,blue,green = [],[],[] #Defined as list so that we can divide by number of items when calculating mean.

            lower_x_bound, upper_x_bound = x-mask_range, (x+mask_range+1)#+1 Because range in for loop does not include largest number
            lower_y_bound, upper_y_bound = y-mask_range, (y+mask_range+1)#+1 Because range in for loop does not include largest number

            if lower_x_bound < 0: #For when mask is out of range in edge cases, ignore those "outer" pixels.
            	lower_x_bound = 0
            if upper_x_bound >= width:
            	upper_x_bound = width
            if lower_y_bound < 0:
            	lower_y_bound = 0
            if upper_y_bound >= height:
            	upper_y_bound = height




            for xi in range(lower_x_bound, upper_x_bound):
            	for yi in range(lower_y_bound, upper_y_bound):
            		pxl2 = original_img.getpixel((xi,yi)) #Iterate over original image
            		red.append(pxl2[0]) #red is pixel[0]
            		green.append(pxl2[1]) # green is pixel[1]
            		blue.append(pxl2[2]) # blue is pixel[2]
            		#raise Exception "The image is getting Blurred, yene fekir"

            red_average = int(np.mean(red))
            green_average = int(np.mean(green))
            blue_average = int(np.mean(blue)) #Setting each color to the mean of pixels aroujnd set mask size.



            img.putpixel((x,y),(red_average,green_average,blue_average)) #Modify copy of image

    base_name = '.'.join(fptr.split(".")[0:-1])
    fptr_2 = base_name + "_blurred.png"
    img.save(fptr_2)
    fptr_2 = base_name + "_original.png" #Setting new name of file
    original_img.save(fptr_2)

    print("Your image has been blurred!")


def set_luminance(fptr, l_val):
	original_img = Image.open(fptr).convert("RGB")
	img = Image.open(fptr).convert("RGB")
	width, height = img.size
	img_lum = get_luminance(img)
	luminance_factor = (l_val/img_lum)

	print(f"Your picture's luminance is {img_lum}!")
	

	for x in range(width):
		for y in range(height):
			pxl = original_img.getpixel((x,y)) #Get pixels from original image
			lum_pxl = get_pxl_luminance(pxl)

			if lum_pxl <= 0:
				lum_pxl = 1 #Do not want to divide by 0
				pxl = (1,1,1)#Reset pixel
			
			r = pxl[0] if pxl[0] else 1
			g = pxl[1] if pxl[1] else 1
			b = pxl[2] if pxl[2] else 1# if any of the pixel colors are 0, set it to 1. This is so that they can still be brightened by the luminance factor

			
			red = int((r)* luminance_factor) #Multiply all colors of a pixel by a constant
			green = int((g) * luminance_factor) 
			blue = int((b) * luminance_factor)
			new_pxl = (red, green, blue)
			img.putpixel((x,y),new_pxl)

	print(f"Now your picture's brightness is {get_luminance(img)}!")

	base_name = '.'.join(fptr.split(".")[0:-1])
	fptr_2 = base_name + "brightened_or_dimmed.png" #setting new name of blurred image
	img.save(fptr_2)
	fptr_2 = base_name + "_original.png"
	original_img.save(fptr_2) #saving image


def get_pxl_luminance(pxl):
    #Calculates the luminance of a single pixel through an equation derived from human perception of visible light.
    #You would input your (1x3)-tuple representing a pixel.

	l_pxl = 0.299 * pxl[0] + 0.587 * pxl[1] + 0.114 * pxl[2] #pxl 0,1 and 2 refer to colors red,green and blue respectively.
	return l_pxl



def get_luminance(img):
	width, height = img.size
	l_val = 0 #Initialize luminance value of image

	for x in range(width):
		for y in range(height): #Iterate over every pixel
			pxl = img.getpixel((x,y))
			l_val += get_pxl_luminance(pxl) #Add all luminances of all pizels together

	l_avg = l_val/(width*height) #Divide y number of pixels to get average luminance of image
	return l_avg


def array_image(fptr, normalize = True, verbose=True,transparent = False, hsv=False):
	'''Converts image to an array of type defined
	as float, float32, integer; 8int.
	array is accessible as:
	array[height][width][color:red,gree,or blue] OR:
	array[rows][columns][colorsRGB]
	'''
	if transparent:
		image = Image.open(fptr).convert("RGBA") #Image object
		width, height = image.size

	elif hsv:
		image = Image.open(fptr).convert("HSV") #Image object
		width, height = image.size

	else:
		image = Image.open(fptr).convert("RGB") #Image object
		width, height = image.size
	# for x in range(width):
	# 	for y in range(height):
	# 		image.putpixel((x,y),(float(x)/255,float(y)/255))


	
		
	array_i = np.asarray(image)
	
	array_i = array_i.astype(np.float32) # Ensures image is not read-only so it's normalizeable
	if normalize:
		array_i /= 255
	if verbose:
		print(f"Image path is {fptr}")
		print(f"IMAGE size : {width} x {height}")	
		print((array_i[0][0][0]))
		print(f"Array shape: {array_i.shape}")
	return array_i

def pixel_reduce(image_array, mask):
	new_image = transform.resize(image_array, (image_array.shape[0] // int(mask), \
                                                image_array.shape[1] // int(mask)), \
                                    anti_aliasing=True)
	return new_image
def image_array_resize(image_array,size):
	new_image = transform.resize(image_array,size)
	return new_image
@adapt_rgb(each_channel)
def sobel_colour(image_array, random=True):
	'''
	Finds edges but retains color, cool.
	'''
	return filters.sobel(image_array)

def centralize_image(image_array, show=False):
	'''
	Takes image, finds object in image by: 1) finding
	edges based on intensity gradient in picture, 2)
	finds coordinates of object and 3) returns "centralized"
	image. The different stages of the processing can be
	shown by setting show=True.
	**Parameters**
	'''
	gray_image = rgb2grey(image_array)
	if show:
		plt.subplot(3,3,1)
		plt.imshow(gray_image, cmap = plt.cm.gray)
	edges = filters.sobel(gray_image) #Get edges, faster to do it when gray
	if show:
		plt.subplot(3,3,2)
		plt.imshow(edges, cmap = plt.cm.gray)

	binary_image = filters.threshold_local(edges, block_size=3) #Turn to binary for filling holes
	if show:
		plt.subplot(3,3,3)
		plt.imshow(binary_image, cmap = plt.cm.gray)

	holes_filled = ndi.binary_fill_holes(binary_image) # fill holes
	if show:
		plt.subplot(3,3,4)
		plt.imshow(holes_filled, cmap = plt.cm.gray)

	selected_image = label(holes_filled) # Most empty space is removed
	if show:
		plt.subplot(3,3,5)
		plt.imshow(selected_image, cmap = plt.cm.gray)
	coordinate_slices = ndi.measurements.find_objects(selected_image)[0]
	# Returns Slice of min x,y and max x,y can be directly inputed into image_array
	
	new_image = image_array[coordinate_slices]
	if show:
		plt.subplot(3,3,6)
		plt.imshow(new_image)
		plt.show()
	# if show:
	# 	images = [image_array, gray_image,
	# 				edges, binary_image, holes_filled,
	# 				selected_image, new_image]
	# 	for i in range(len(images)):
	# 		plt.subplot(, 3, i+1)
	# 		plt.imshow(images[i])
	# 	plt.show()
	new_image_r = image_array_resize(new_image, (100,100,3))
	return new_image_r



if __name__ == "__main__":
    # fptr = "dog.jpg" #If your image is called "dog.jpg"
    # blur(fptr, mask=7) #Blur with mask of 7
    # set_luminance(fptr, 150.0) #Set luminance to ~150
	# print("Ok!")
	# array_image1 = array_image("Data/pokemon-images-dataset-by-type/all/Arceus.png")
	# #using forward slahses for paths is better
	# print(array_image1.shape)
	# plt.subplot(1,3,1)

	# plt.imshow(array_image1)

	# array_image_less_pixels = pixel_reduce(array_image1, mask=2)
	# plt.subplot(1,3,2)
	# plt.imshow(array_image_less_pixels)

	# array_image_borders = sobel_colour(array_image1)
	# plt.subplot(1,3,3)
	# plt.imshow(array_image_borders)
	# plt.show()

	#For large database:
	database_dir = "C:/Users/Nelson/Desktop/Software carp/Final project/Data/Train_test_Database"
	file_name_X = "Database_6000_X_images.npy"
	relative_path1 = os.path.join(database_dir,file_name_X)

	X = np.load(relative_path1, allow_pickle=True)
	centralize_image(X[5000], show= True)

	garch = pltcolors.rgb_to_hsv(X[5000]) #Not normalized
	garch = image_array_resize(garch, (64,64,3))
	plt.imshow(garch)
	plt.show()
	centralize_image(garch, show=True)

