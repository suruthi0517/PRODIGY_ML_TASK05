import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load and preprocess calorie data (CSV format)
def load_calorie_data(calorie_data_path):
    try:
        calorie_data = pd.read_csv(calorie_data_path)
        print("Calorie data loaded successfully.")
        
        # Normalize the 'food_item' keys to lowercase and remove whitespace
        calorie_data['food_item'] = calorie_data['food_item'].str.lower().str.strip()
        return calorie_data.set_index("food_item")["calories"].to_dict()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {calorie_data_path}")
        return {}
    except KeyError:
        print("Error: CSV file should contain 'food_item' and 'calories' columns.")
        return {}

# Step 2: Load and preprocess image dataset
def load_dataset(dataset_path, image_size=(64, 64)):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder not found at {dataset_path}")
        return np.array(images), np.array(labels), label_map

    for category_path, _, files in os.walk(dataset_path):
        if not files:
            continue
        category_name = os.path.basename(category_path).lower().strip()
        label_map[label_counter] = category_name
        print(f"Processing category: {category_name}")
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(label_counter)
        label_counter += 1

    return np.array(images), np.array(labels), label_map

# Step 3: Define pretrained model
def build_pretrained_model(num_classes, image_size=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_size)
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_test, y_test),
                        epochs=10)
    return history

# Step 5: Predict and visualize results
def predict_image(image_path, model, label_map, calorie_data, image_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize the image

        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)
        food_item = label_map[predicted_label].lower().strip()

        # Retrieve calorie information
        calorie_count = calorie_data.get(food_item, "Unknown")

        print(f"Predicted Food: {food_item}, Calories: {calorie_count}")
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title(f"Food: {food_item}\nCalories: {calorie_count} kcal")
        plt.axis('off')
        plt.show()
    else:
        print("Error: Unable to load the image.")

# Main Function
def main():
    # File paths
    calorie_data_path ="D:/ML_INTERNSHIP/Task-5/Task05_dataset/calorie_data.csv"
    dataset_path = "D:/ML_INTERNSHIP/Task-5/Task05_dataset/food_images"

    # Load data
    calorie_data = load_calorie_data(calorie_data_path)
    images, labels, label_map = load_dataset(dataset_path, image_size=(64, 64))

    if len(images) == 0 or len(labels) == 0:
        print("Error: Dataset is empty or improperly formatted.")
        return

    images = images / 255.0  # Normalize images

    # Debugging: Print label map and calorie data keys
    print("Label Map:", label_map)
    print("Calorie Data Keys:", list(calorie_data.keys()))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train model
    model = build_pretrained_model(num_classes=len(label_map), image_size=(64, 64, 3))
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Save the model
    model.save('food_calorie_estimator_model.h5')
    print("Model training complete and saved.")

    # Predict using a sample image
    test_image_path = "D:/ML_INTERNSHIP/Task-5/Task05_dataset/food_images/pizza/pizza_2.jpg"  # Provide a valid image path
    if os.path.exists(test_image_path):
        predict_image(test_image_path, model, label_map, calorie_data, image_size=(64, 64))
    else:
        print("Error: The specified image path does not exist.")

if __name__ == "__main__":
    main()
