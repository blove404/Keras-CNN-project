#Keras CNN project on mammals and birds, Telling the difference in images between the two categories, it analyzes the data provided a little over 
#50,000 images and then takes an image uploaded in BirdorMammal and then determines if the image is a bird or mammal, specifically Animal.png
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Set up directories
data_dir = 'CS3400 project/Dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

# Define image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

# Create the directory if it doesn't exist
model_dir = 'CS3400 project'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model weights
model_weights_path = os.path.join(model_dir, 'model_weights.weights.h5')
print(f"Saving model weights to: {model_weights_path}")
model.save_weights(model_weights_path)

# Save the entire model
model_path = os.path.join(model_dir, 'my_model.keras')
print(f"Saving entire model to: {model_path}")
model.save(model_path, overwrite=True)

# Evaluation code
def classify_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Load the trained weights
    model.load_weights(model_weights_path)

    # Make the prediction
    prediction = model.predict(img_array)[0][0]

    # Determine the class
    if prediction > 0.5:
        return 'Mammal'
    else:
        return 'Bird'

# Example usage
image_path = 'CS3400 project/Birdormammal/Animal.jpg'
predicted_class = classify_image(image_path)
print(f'The image is classified as a {predicted_class}.')