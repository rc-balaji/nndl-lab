import tensorflow as tf
import numpy as np
x_train=np.array([[0,0],[0,1],[1,0],[1,1]])
y_train=np.array([0,0,0,0])
model=tf.keras.Sequential([tf.keras.layers.Dense(1,activation='sigmoid',input_shape=(2,))])
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=6)
x_test=np.array([[0,0],[0,1],[1,0],[1,1]])
predictions=model.predict(x_test)
print('predictions:',predictions)