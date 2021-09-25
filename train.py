# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:24:11 2021

@author: LiangChih
"""
#%% Load package
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def getAbsPathOfFolder(path):
    list_of_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            list_of_files.append(os.path.join(root,file))
    return list_of_files

tf.compat.v1.disable_eager_execution()

#%% Data preparing
# Data path
dogPath = r".\dataset\dog"
catPath = r".\dataset\cat"

# Get data path list
dogFiles = getAbsPathOfFolder(dogPath)
catFiles = getAbsPathOfFolder(catPath)

#%% Show Image
"""
# Read JPEG or PNG or GIF image from file
image = plt.imread(dogFiles[0])
# Reshape image
reshaped_image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
# Subtract off the mean and divide by the variance of the pixels.
reshaped_image = tf.image.per_image_standardization(reshaped_image)
# Resize image to 224*224 and zero padding
crop_resize_image = tf.image.resize_with_pad(reshaped_image, 224, 224)
sess = tf.compat.v1.Session()
b = crop_resize_image.eval(session = sess)
plt.imshow(b[0].astype('uint8'))
plt.show()
"""

#%% Image Processing
def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize image to 224*224 and zero padding
    crop_resize_image = tf.image.resize_with_pad(image, 224, 224)
    
    final_image = tf.keras.applications.xception.preprocess_input(crop_resize_image)
    
    return final_image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

#%% filenames and lebels
filenames = dogFiles + catFiles
listofzeros = [0] * 110
listofones = [1] * 110
labels = listofzeros + listofones

#%% tensorflow pipline
batch_size = 5
epochs=5

full_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
full_dataset = full_dataset.shuffle(len(filenames))

full_dataset = full_dataset.repeat(epochs).map(parse_function).batch(batch_size).prefetch(1)


#full_dataset = full_dataset.map(parse_function)
#full_dataset = full_dataset.map(train_preprocess)
#full_dataset = full_dataset.batch(batch_size)
#full_dataset = full_dataset.prefetch(1)

#%%
dataset_size = len(labels)

train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = int(0.2 * dataset_size)

train_set = full_dataset.take(train_size)
test_set = full_dataset.skip(train_size)
valid_set = test_set.skip(val_size)
test_set = test_set.take(test_size)

#%% Load model
n_classes = 2
base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)


for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.6 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.2 * dataset_size / batch_size),
                    epochs=5)

"""
#%% tensorflow pipline
batch_size = 5
epochs=40

full_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
full_dataset = full_dataset.shuffle(len(filenames))

full_dataset = full_dataset.repeat(epochs).map(parse_function).batch(batch_size).prefetch(1)


#full_dataset = full_dataset.map(parse_function)
#full_dataset = full_dataset.map(train_preprocess)
#full_dataset = full_dataset.batch(batch_size)
#full_dataset = full_dataset.prefetch(1)

#%%
dataset_size = len(labels)

train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = int(0.2 * dataset_size)

train_set = full_dataset.take(train_size)
test_set = full_dataset.skip(train_size)
valid_set = test_set.skip(val_size)
test_set = test_set.take(test_size)


#%%
for layer in base_model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.6 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.2 * dataset_size / batch_size),
                    epochs=40)

"""




























