import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

train, info = tfds.load("eurosat/rgb", split="train[:80%]", with_info=True)
test = tfds.load("eurosat/rgb", split="train[80%:90%]")
validation = tfds.load("eurosat/rgb", split="train[90%:]")

NUM_EPOCHS = 5
BATCH_SIZE = 128
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = int(info.splits["train"].num_examples * 0.8)//BATCH_SIZE
VALIDATION_STEPS = int(info.splits["train"].num_examples * 0.1)//BATCH_SIZE
IMAGE_SHAPE = [180, 180]
num_classes = 10

@tf.function
def prepare_training_data(datapoint):
    input_image = tf.image.resize(datapoint["image"], IMAGE_SHAPE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_flip_left_right(input_image)
        input_image = tf.image.random_flip_up_down(input_image)
        input_image = tf.image.random_brightness(input_image, max_delta=0.3)
        input_image = tf.image.random_saturation(input_image, lower=0.75, upper=1.5)
        input_image = tf.image.random_contrast(input_image, lower=0.75, upper=1.5)

    input_image = preprocess_input(input_image)
    return input_image, datapoint["label"]

def prepare_validation_data(datapoint):
    input_image = tf.image.resize(datapoint["image"], IMAGE_SHAPE)
    input_image = preprocess_input(input_image)

    return input_image, datapoint["label"]

train = train.map(prepare_training_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation = validation.map(prepare_validation_data)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation.batch(BATCH_SIZE)

resnet = ResNet50(input_shape=IMAGE_SHAPE + [3], weights='imagenet', include_top=False)

for layer in resnet.layers:
    layer.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
prediction = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=resnet.input, outputs=prediction)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS
)

test = test.map(prepare_validation_data)
test_dataset = test.batch(BATCH_SIZE)

for image, label in tfds.as_numpy(test_dataset.take(1)):
    choice = np.random.randint(len(label))
    image = np.expand_dims(image[choice], axis=0)
    pred_label = np.argmax(model.predict(image))
    print(pred_label, label[choice])
