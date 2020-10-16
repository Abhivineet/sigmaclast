from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import keras

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(224,224,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()


folder_path = '/Users/abhivineet/PycharmProjects/sigmaclast/data2'
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=180, width_shift_range=0.3, height_shift_range=0.3, rescale=1./255, zoom_range=0.4, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')

train_generator=train_datagen.flow_from_directory(folder_path,
                                                 target_size=(224,224),
                                                 color_mode='grayscale',
                                                 batch_size=16,
                                                 class_mode='categorical',
                                                 shuffle=True)

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=20)
model.save('model_occamConv.h5')