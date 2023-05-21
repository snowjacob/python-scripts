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

#Hyperparamters
epochs = 5

lr = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07
weight_decay = 0.0

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
    layers.Dense(10, activation=PReLU()),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay = weight_decay)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=epochs
)