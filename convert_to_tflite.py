import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('dog_cat_model.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('dog_cat_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Conversion complete! Saved as dog_cat_model.tflite")
