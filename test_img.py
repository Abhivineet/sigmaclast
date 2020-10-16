from keras.preprocessing import image
from keras.models import load_model
import keras
import numpy as np

model = load_model('model_transResNet50.h5')
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

img = image.load_img('test.jpg', target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

val = model.predict(img)
print(val)