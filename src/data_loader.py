# src/data_loader.py

import tensorflow as tf

IMAGE_SIZE = 256
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def get_dataset(data_dir):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    return dataset, dataset.class_names

def split_dataset(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    assert train_split + val_split + test_split == 1.0

    dataset_size = dataset.cardinality().numpy()
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    dataset = dataset.shuffle(10000, seed=12)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    return train_ds, val_ds, test_ds

def prepare_datasets(train_ds, val_ds, test_ds):

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds

