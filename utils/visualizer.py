# utils/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_training_curves(history, save_path=None):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

def display_predictions(model, test_ds, class_names, num_images=9):
    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class, confidence = predict_single_image(model, images[i], class_names)
            actual_class = class_names[labels[i]]
            plt.title(f"Actual: {actual_class}\nPred: {predicted_class}\nConf: {confidence}%")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

def predict_single_image(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

