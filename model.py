import csv

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generator(samples, batch_size=32):
    """
    Generator for training the network without loading all samples at once.  
    
    :param samples: Images from the cameras
    :param batch_size: number of images to be processed in each batch
    :return: the generator yields the processed images and the stear and of each image
    """
    num_samples = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images_gen = []
            measurements_gen = []

            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]

                    file_name = source_path.split('/')[-1]
                    image_path = './data_two/IMG/' + file_name
                    image = cv2.imread(image_path)
                    measurement = float(line[STEAR_POSITION])

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = augment_brightness_camera_images(image)
                    # image = add_random_shadow(image)

                    images_gen.append(image)
                    images_gen.append(cv2.flip(image, 1))

                    # Getting all the three cameras images.
                    if i == 0:
                        measurements_gen.append(measurement)
                        measurements_gen.append(measurement * -1.0)

                    if i == LEFT_SIDE:
                        measurement = measurement + CORRECTION_ANGLE
                        measurements_gen.append(measurement)
                        measurements_gen.append(measurement * -1.0)

                    if i == RIGHT_SIDE:
                        measurement = measurement - CORRECTION_ANGLE
                        measurements_gen.append(measurement)
                        measurements_gen.append(measurement * -1.0)

            batch_features = np.array(images_gen)
            batch_labels = np.array(measurements_gen)

            yield shuffle(batch_features, batch_labels)


def le_net():
    """
    LeNet architecture. Used to train try to solve the problem in the first tries. 
    :return: 
    """
    le_net_model = Sequential()
    le_net_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    le_net_model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    le_net_model.add(Convolution2D(6, 5, 5, activation="relu"))
    le_net_model.add(MaxPooling2D())
    le_net_model.add(Convolution2D(6, 5, 5, activation="relu"))
    le_net_model.add(MaxPooling2D())
    le_net_model.add(Flatten())
    le_net_model.add(Dense(128))
    le_net_model.add(Dense(84))
    le_net_model.add(Dense(1))

    return le_net_model


def nvidia_net():
    """
    NvidiaNet based in the tutorials of Udacity
    :return: 
    """
    nvidia_model = Sequential()
    nvidia_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    nvidia_model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    nvidia_model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    nvidia_model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    nvidia_model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    nvidia_model.add(Convolution2D(64, 3, 3, activation="relu"))
    nvidia_model.add(Convolution2D(64, 3, 3, activation="relu"))
    nvidia_model.add(Flatten())
    nvidia_model.add(Dropout(0.5))
    nvidia_model.add(Dense(100))
    nvidia_model.add(Dense(50))
    nvidia_model.add(Dense(10))
    nvidia_model.add(Dropout(0.5))
    nvidia_model.add(Dense(1))

    return nvidia_model


def augment_brightness_camera_images(image):
    """
    Data augmentation technique taken from this tutorial: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    This method helped a lot the car to drive correctly. 
    :param image: 
    :return: 
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def plot_history(history_obj):
    """
    Method for ploting a chart showing the loss of the training and validation set
    :param history_obj: 
    :return: 
    """
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


lines = []
STEAR_POSITION = 3
EPOCHS = 1
CORRECTION_ANGLE = 0.25
LEFT_SIDE = 1
RIGHT_SIDE = 2
BATCH_SIZE = 64


# Instead of one big pool of training data, I split it in many specific trainings focused
# in different part of the driving process. I drove 3 laps in the correct way, 1 in the backwards
# way and 2 times I just recorded the car making curves.
# This helped to validate witch way of driving helped the car to generalize better. Some approached didn't work
# like: - Recording the car recovering from the sides of the road made the model worse.
#       - Recording the car on the second track made is worse. The car learned that it should follow a center line,
# so it kept changing between the two side lines.

with open("./data_two/driving_log_center_one.csv") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)

with open("./data_two/driving_log_center_back_one.csv") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)

with open("./data_two/driving_log_center_two.csv") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)

with open("./data_two/driving_log_curves.csv") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)

with open("./data_two/driving_log_center_three.csv") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)

with open("./data_two/driving_log_curves_two.csv") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = nvidia_net()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=EPOCHS,
                    verbose=1)

print("Saving results")
model.save('model_one.h5')
print("Saved")

plot_history(history_object)

