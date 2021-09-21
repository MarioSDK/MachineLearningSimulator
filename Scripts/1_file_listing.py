import numpy as np
import glob
import cv2

one_side = ["angle"]
three_sides = ["left", "straight", "right"]
seven_sides = [
	"left_3",
	"left_2",
	"left_1",
	"straight",
	"right_1",
	"right_2",
	"right_3"]
eleven_sides = [
	"left_25",
	"left_20",
	"left_15",
	"left_10",
	"left_05",
	"straight",
	"right_05",
	"right_10",
	"right_15",
	"right_20",
	"right_25"]

def reduce_image(image):
	crop_image  = np.array(image[57:136, 0:320])
	gray_image  = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
	array_image = np.asarray(gray_image)
	
	row    = np.array(np.arange(0,  80, 5), dtype=np.intp)
	column = np.array(np.arange(0, 320, 5), dtype=np.intp)
	
	pixels = np.copy(array_image)
	pixels = pixels[row[:, np.newaxis], column]
	pixels = pixels.flatten()
	
	return pixels

def prepare_side_images(folder_path, sides, limit):
	for side in sides:

		filepaths = glob.glob(folder_path + "/" + side + "/*.jpg")
		filename = folder_path + "/imrows_" + side + ".csv"
		newfile = open(filename, "w+")
		
		print("Writing file " + filename + "...")
		index = 0

		for filepath in filepaths:
			if index == limit:
				break
			
			# 320 x 160
			image = cv2.imread(filepath)
			
			# 320 x 80
			pixels = reduce_image(image)
			
			string = ','.join(str(int(p)) for p in pixels)
			newfile.write(string)
			newfile.write("\n")
			
			index += 1 
		
		print("Done!")
		newfile.close()



prepare_side_images("images_three_sides",  three_sides,  limit=300)
#prepare_side_images("images_eleven_sides", eleven_sides, limit=200)
#prepare_side_images("images_seven_sides", seven_sides, limit=400)
