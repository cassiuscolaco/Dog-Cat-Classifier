import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# Build a simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# Save the trained model to a file
model.save('dog_cat_model.h5')
print("✅ Model trained and saved as dog_cat_model.h5")
