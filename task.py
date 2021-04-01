from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

HEIGHT = 300
WIDTH = 300

base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


class_list = ["Cat", "Dog", "Horse"]
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

from tensorflow.keras.optimizers import Adam

NUM_EPOCHS = 10
BATCH_SIZE = 8
# num_train_images = 300

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "C:\\Users\\ASUS\\Desktop\\python_prac\\Onports\\train"
HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8
# Creates an instance of an ImageDataGenerator called train_datagen, train_generator,
# train_datagen.flow_from_directory

# splits data into training and testing(validation) sets
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.25
)

# Training data
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    subset='training')

# Validation data
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                         target_size=(HEIGHT, WIDTH),
                                                         batch_size=BATCH_SIZE,
                                                         subset='validation')

finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                             steps_per_epoch=469 // BATCH_SIZE,
                             shuffle=True)

test = finetune_model.evaluate_generator(validation_generator, steps=155 // BATCH_SIZE)
print('Validation Loss : ', test[0])
print('Validation Accuracy : ', test[1])

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
#image_path = "C:\\Users\\ASUS\\Desktop\\python_prac\\Onports\\images\\dog.jpg"

def conv(image_path):
  imgsize = 300
  img_array = cv2.imread(image_path)
  new_array = cv2.resize(img_array, (300, 300))
  return new_array.reshape(-1, imgsize, imgsize, 3)

image = conv(image_path)
# print(image.shape)
pred = finetune_model.predict([image])
classes = np.argmax(pred, axis = 1)
# print(classes)
# print(class_list[int(classes[0])])

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_colors(image, number_of_colors):
    modified_image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] / 255 for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i] * 255) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] * 255 for i in counts.keys()]


    plt.figure(figsize=(8, 6))
    plt.title(class_list[int(classes[0])])
    plt.pie(counts.values(), labels=hex_colors, colors=ordered_colors, shadow=True)
    plt.show()

    return rgb_colors

# get_colors(get_image(image_path), 5, True)


# GUI
from tkinter import *
root = Tk()

root.geometry('400x200')
root.minsize(width=400, height=200)

root.title('Animal Classification & Color Detection ')

Imagepath = Label(root, text = 'Imagepath')
Colorshades = Label(root, text = 'Colorshades')


Imagepath.grid() # pack
Colorshades.grid(row=1) # pack

# BooleanVar, StringVar, DoubleVar, IntVar
Imagepath = StringVar()
Colorshades = IntVar()

e1 = Entry(root, textvariable = Imagepath)
e2 = Entry(root, textvariable = Colorshades)

e1.grid(row = 0, column = 1)# pack
e2.grid(row = 1, column = 1)# pack


def fun():
    global Imagepath, Colorshades
    return get_colors(get_image(Imagepath.get()), Colorshades.get())

b1 = Button(text = 'Run', command = fun)
b1.grid(column = 1)

root.mainloop()

