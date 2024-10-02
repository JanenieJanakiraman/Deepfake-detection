


pip install tensorflow keras opencv-python matplotlib numpy pandas scikit-learn





import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#accessing the dataset
base_path = '/Users/janeniej/Downloads/Dataset'
train_path = os.path.join(base_path, '/Users/janeniej/Downloads/Dataset/Train')
test_path = os.path.join(base_path, '/Users/janeniej/Downloads/Dataset/Test')
validate_path = os.path.join(base_path, '/Users/janeniej/Downloads/Dataset/Validation')

# here, I am loading the images
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['real', 'fake']:
        folder_path = os.path.join(folder, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Resize to your preferred size
                images.append(img)
                labels.append(0 if label == 'real' else 1)  # Assign labels
    return np.array(images), np.array(labels)

# loading the training, test validation images.
X_train, y_train = load_images_from_folder(train_path)
X_val, y_val = load_images_from_folder(validate_path)
X_test, y_test = load_images_from_folder(test_path)

# Normaliing images
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0





datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=20)




test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")



model.save('deepfake_detection_model.h5')

