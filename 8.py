from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers

# Load pretrained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_data, train_labels, batch_size=32, epochs=10)

# Evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print('Accuracy:', test_accuracy)