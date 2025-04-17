# src/evaluate.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from utils.visualizer import display_predictions

from data_loader import get_dataset, split_dataset, prepare_datasets

DATA_DIR = "data/raw"
MODEL = "2"
MODEL_PATH = f"models/{MODEL}/model.h5"  # Update if using timestamped file
NUM_IMAGES = 9  # images to visualize

def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch
    predictions = model.predict(img_array, verbose=0)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

def main():
    print("[INFO] Loading dataset...")
    dataset, class_names = get_dataset(DATA_DIR)
    _, _, test_ds = split_dataset(dataset)
    _, _, test_ds = prepare_datasets(_, _, test_ds)

    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Making predictions...")
    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(NUM_IMAGES):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class, confidence = predict(model, images[i].numpy(), class_names)
            actual_class = class_names[labels[i]]

            plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence}%")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")  # Not actual confusion matrix but visual prediction
    plt.show()

    display_predictions(model, test_ds, class_names)

if __name__ == "__main__":
    main()

