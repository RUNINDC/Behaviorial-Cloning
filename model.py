import csv
import cv2
import keras
import numpy as np
import os
import sklearn

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle


### basic setup
# constants
BASE_FOLDER = "./data/"
RAW_DATA_PATH = BASE_FOLDER + ""
LOGFILE_NAME = "driving_log.csv"

# file related
lines = []

# training paramters
nb_epochs = 25

# side camera parameters
correction_factor = 0.17

# feedback
print("basic setup completed")

### helper functions
# adopted from example in classroom
def generator(lines_, batch_size=32):
	num_samples = len(lines_)
	while 1:
		shuffle(lines_)
		for offset in range(0, num_samples, batch_size):
			batch_samples = lines_[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				# read data
				center_name  = RAW_DATA_PATH+'IMG/'+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(center_name)
				center_angle = float(batch_sample[3])
				
				left_name = RAW_DATA_PATH+'IMG/'+batch_sample[1].split('/')[-1]
				left_image = cv2.imread(left_name)
				left_angle = center_angle+correction_factor

				right_name = RAW_DATA_PATH+'IMG/'+batch_sample[2].split('/')[-1]
				right_image = cv2.imread(right_name)
				right_angle = center_angle-correction_factor

				# store original data
				images.append(center_image)
				images.append(left_image)
				images.append(right_image)
				angles.append(center_angle)
				angles.append(left_angle)
				angles.append(right_angle)

				# mirror data (flip)
				flipped_center_image = cv2.flip(center_image, 1)
				flipped_left_image = cv2.flip(left_image, 1)
				flipped_right_image = cv2.flip(right_image, 1)
				flipped_center_angle = center_angle * -1
				flipped_left_angle = left_angle * -1
				flipped_right_angle = right_angle * -1
				# store flipped data
				images.append(flipped_center_image)
				images.append(flipped_left_image)
				images.append(flipped_right_image)
				angles.append(flipped_center_angle)
				angles.append(flipped_left_angle)
				angles.append(flipped_right_angle)
				
			# convert to numpy array
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def parse_driving_log(path_to_folder):
	temp_lines = []

	# feedback
	print("started parsing driving log in folder " + path_to_folder)

	# load data from csv
	with open(path_to_folder + LOGFILE_NAME) as csvfile:
		reader = csv.reader(csvfile)
	
		for line in reader:
			temp_lines.append(line)

	#feedback
	print("finished parsing driving log")

	#return result
	return temp_lines

def split_parsed_lines_to_sets(input_lines, test_size=0.1, nb_shuffles=1):
	num_lines = len(input_lines)
	split_index = int(num_lines*(1-test_size))
	
	parsed_lines = input_lines
		
	for i in range(nb_shuffles):
		shuffle(parsed_lines)
	
	train_lines = parsed_lines[0:split_index]
	valid_lines = parsed_lines[split_index:num_lines]

	return (train_lines, valid_lines)

# feedback
print("finished loading helper functions")


### load data and prepare
# parse csv
print("parsing driveing_log...")
lines = parse_driving_log(RAW_DATA_PATH)
# split data
print("splitting driving data...")
train_lines_tmp , valid_lines_tmp = split_parsed_lines_to_sets(lines, 0.2, 1)

# calculate correct factor for number of samples per epoch
# assuming batch size for generator is set to 32
final_batch_size = 32*6

train_modulo = len(train_lines_tmp)%final_batch_size
valid_modulo = len(valid_lines_tmp)%final_batch_size

train_lines = train_lines_tmp[0:len(train_lines_tmp)-train_modulo]
valid_lines = valid_lines_tmp[0:len(valid_lines_tmp)-valid_modulo]


# prepare generators
print("configuring generators")
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(valid_lines, batch_size=32)


### Keras model
print("setting up Keras model...")
# Setup
model = Sequential()

# normalize input image
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# crop input
model.add(Cropping2D(cropping=((70,25), (0,0))))

# first convolutional layer
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))
# second convolutional layer
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))
# third convolutional layer
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))
# fourth convolutional layer
model.add(Convolution2D(64, 3, 3))
model.add(ELU())
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(ELU())
model.add(Dropout(0.5))

# flatten layer
model.add(Flatten())

# first fully connected layer
model.add(Dense(100))
# second fully connected layer
model.add(Dense(50))
# third fully connected layer
model.add(Dense(10))
# output layer of the predicted steering angle
model.add(Dense(1))


### compile model
print("compiling Keras model...")
model.compile(optimizer='adam', loss='mse')


### train model
print("starting to train...")
history = model.fit_generator(train_generator, samples_per_epoch=len(train_lines)*3, validation_data=validation_generator, nb_val_samples=len(valid_lines)*3, nb_epoch=nb_epochs)


### store model
print("saving net")
model.save('model.h5')
