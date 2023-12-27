
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Use ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

 # Load data and split into train, validation, and test sets
train_generator = train_datagen.flow_from_directory('chest_xray/train', target_size=(224, 224), batch_size=32, class_mode='binary')
val_generator = test_datagen.flow_from_directory('chest_xray/val', target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory('chest_xray/test', target_size=(224, 224), batch_size=32, class_mode='binary')

# # Define the model (using a simple CNN for demonstration purposes)
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# model.save('chest_xray_model.h5')
model = load_model('chest_xray_model.h5')
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc*100}')

# Make predictions on the test set
predictions = model.predict(test_generator)
predicted_classes = np.round(predictions)

# Display confusion matrix and classification report
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


