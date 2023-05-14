import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import PReLU
from keras import Sequential

# Load the data

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/jacob/source/repos/Fire_Prediction/Fire_Prediction/archive/train',
    labels='inferred',
    label_mode='int',
    class_names=['nowildfire', 'wildfire'],
    color_mode='rgb',
    batch_size=20,
    image_size=(350, 350),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',
    interpolation='bilinear'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/jacob/source/repos/Fire_Prediction/Fire_Prediction/archive/train',
    labels='inferred',
    label_mode='int',
    class_names=['nowildfire', 'wildfire'],
    color_mode='rgb',
    batch_size = 20,
    image_size=(350, 350),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation',
    interpolation='bilinear'
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Simple sequential model in order to get the probability of either a fire or no fire
model = Sequential([
    layers.Rescaling(1./255, input_shape=(350, 350, 3)),
    layers.Conv2D(10, 3, padding='same', activation=PReLU()),
    layers.MaxPooling2D(),
    layers.Conv2D(12, 3, padding='same', activation=PReLU()),
    layers.MaxPooling2D(), 
    layers.Conv2D(16, 3, padding='same', activation=PReLU()),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(20, activation=PReLU()),
    layers.Dense(10, activation='sigmoid')
    layers.Dense(activation='softmax')
])

model.compile(optimizer='adagrad',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 3

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=epochs
)