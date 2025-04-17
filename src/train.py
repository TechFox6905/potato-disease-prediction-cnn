# src/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current script directory:", current_dir)

import matplotlib.pyplot as plt
from utils.visualizer import plot_training_curves
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import get_dataset, split_dataset, prepare_datasets
from model import build_model

DATA_DIR = "data/raw"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
EPOCHS = 20

def plot_metrics(history, output_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(history.history["accuracy"]))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "accuracy_loss_curves.png"))
    plt.show()

def get_new_model_version(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    existing_versions = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
    return max(existing_versions) + 1 if existing_versions else 1

def main():
    print("[INFO] Loading dataset...")
    dataset, class_names = get_dataset(DATA_DIR)
    n_classes = len(class_names)

    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    print("[INFO] Building model...")
    model = build_model(n_classes)

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

    # Define the ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Monitors the validation loss
        factor=0.2,  # Reduce learning rate by this factor
        patience=5,  # Wait for 5 epochs without improvement
        min_lr=1e-6  # Minimum learning rate
    )

    print("[INFO] Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    print("[INFO] Evaluating model...")
    scores = model.evaluate(test_ds)
    print(f"Test Accuracy: {round(scores[1] * 100, 2)}%")

    print("[INFO] Saving model and plots...")
    model_version = get_new_model_version(MODELS_DIR)
    version_path = os.path.join(MODELS_DIR, str(model_version))
    os.makedirs(version_path, exist_ok=True)
    model.save(os.path.join(version_path, "model.h5"))

    print(f"[INFO] Model saved to: {version_path}/model.h5")

    plot_metrics(history, OUTPUTS_DIR)
    plot_training_curves(history, save_path="outputs/accuracy_loss_curves.png")

if __name__ == "__main__":
    main()

