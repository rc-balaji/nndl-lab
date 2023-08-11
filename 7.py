import tensorflow as tf 
from PIL import Image 
import numpy as np 
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.utils import to_categorical 
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
train_subset_size = 5000 
test_subset_size = 1000 
x_train = x_train[:train_subset_size] 
y_train = y_train[:train_subset_size] 
x_test = x_test[:test_subset_size] 
y_test = y_test[:test_subset_size] 
x_train = x_train.astype('float32') / 255.0 
x_test = x_test.astype('float32') / 255.0 
num_classes = 10 
y_train = to_categorical(y_train, num_classes) 
y_test = to_categorical(y_test, num_classes) 

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers:
    layer.trainable = False 
model = Sequential() 
model.add(base_model) 
model.add(GlobalAveragePooling2D()) 
model.add(Dense(256, activation='relu')) 
model.add(Dense(num_classes, activation='softmax')) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test)) 
_, accuracy = model.evaluate(x_test, y_test, verbose=0) 
print('Test accuracy:', accuracy) 
image_path = 'truck.jpg' 
image = Image.open(image_path) 
image = image.resize((32, 32)) 
image = np.array(image) 
image = image.astype('float32') / 255.0 
image = np.expand_dims(image, axis=0) 
predictions = model.predict(image) 
predicted_labels = tf.argmax(predictions, axis=1) 
class_names = [ 
'airplane', 'automobile', 'bird', 'cat', 'deer', 
'dog', 'frog', 'horse', 'ship', 'truck' 
] 
print('Predicted class:', class_names[predicted_labels[0]]) 