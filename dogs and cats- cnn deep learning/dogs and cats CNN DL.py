'''from machine learning A-Z
   coded by trishit nath thakur'''


# Importing the libraries


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# Part 1 - Data Preprocessing


# Preprocessing the Training set


train_datagen = ImageDataGenerator(rescale = 1./255,  # apply feature scaling to each and every single one of your pixels by dividing their value by 255
                                   shear_range = 0.2, # means shear the image by 20%
                                   zoom_range = 0.2,  # means zoom-in and zoom-out by 20%
                                   horizontal_flip = True) # For mirror reflection, I have given horizontal_flip=True

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # size of training set images 
                                                 batch_size = 32,  # batch of images for training 
                                                 class_mode = 'binary') # dataset is binary


# Preprocessing the Test set


test_datagen = ImageDataGenerator(rescale = 1./255)   # apply feature scaling to each and every single one of your pixels by dividing their value by 255

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# Part 2 - Building the CNN


# Initialising the CNN

cnn = tf.keras.models.Sequential()   #sequential class from models module of keras library which belong to tf. we have now a sequence of layers

# Step 1 - Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) 

 # filter is the number of output filters in the convolution
 # kernel_size is specifying the height and width of the 2D convolution window
 # 3 in input_shape because we have colored images

# Step 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # pool size for window size over which to take the maximum

# strides Specifies how far the pooling window moves for each pooling step

# Adding a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening

cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection

cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) # unit which corresponds exactly to the number of neurons

# Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

 # sigmoid act func will give you the probabilities that the binary outcome is one
 # we want to predict a binary variable which can take the value one or zero

# Part 3 - Training the CNN


# Compiling the CNN

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

  # adam optimizer that can perform stochastic gradient descent(update the weights in order to reduce the loss error between your predictions and the real results)

# Training the CNN on the Training set and evaluating it on the Test set

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# Part 4 - Making a single prediction


import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)  #predict method expects 2D array as input 
test_image = np.expand_dims(test_image, axis = 0) # image needs to be in a batch so we add extra dimension for batch

result = cnn.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'

else:
    prediction = 'cat'

print(prediction)