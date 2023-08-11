import numpy as np
from keras.models import Sequential
from keras.layers import Dense

X = np.random.rand(100, 10) 
y = np.random.randint(2, size=(100, 1))

model = Sequential()

model.add(Dense(32, activation='relu', input_dim=10))

model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=5, batch_size=32)

new_data = np.random.rand(10, 10)

predictions = model.predict(new_data)
print(predictions)