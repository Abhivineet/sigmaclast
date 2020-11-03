# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
#
# img = load_img('data/CW/000_a_50.jpg')
# data = img_to_array(img)
# samples = expand_dims(data, 0)
# datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, vertical_flip=True)
# it = datagen.flow(samples, batch_size=1)
# for i in range(9):
#     plt.subplot(330+1+i)
#     batch = it.next()
#     image=batch[0].astype('uint8')
#     plt.imshow(image)
# plt.show()


folder_path = '/Users/abhivineet/PycharmProjects/datasets/sigmaclast/data2'

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


base_model=ResNet50(weights='imagenet',include_top=False)

newOutput=base_model.output
newOutput=GlobalAveragePooling2D()(newOutput)
newOutput=Dense(1024,activation='relu')(newOutput)
newOutput=Dense(1024,activation='relu')(newOutput)
newOutput=Dense(512,activation='relu')(newOutput)

preds=Dense(2,activation='softmax')(newOutput) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)


for i,layer in enumerate(model.layers):
  print(i,layer.name)


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=180, width_shift_range=0.3, height_shift_range=0.3, rescale=1./255, zoom_range=0.4, horizontal_flip=True, vertical_flip=True, fill_mode='reflect', validation_split=0.1)

train_generator=train_datagen.flow_from_directory(folder_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
val_generator = train_datagen.flow_from_directory(folder_path,
                                                  target_size=(224,224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  subset='validation')

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


train_steps=train_generator.n//train_generator.batch_size
val_steps=val_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps,
                    epochs=10,
                    validation_data=val_generator)

# true_classes = val_generator.classes
# class_labels = list(val_generator.class_indices.keys())

# ACCURACY OF 50% LOL
# model.save('model_transResNet50.h5')