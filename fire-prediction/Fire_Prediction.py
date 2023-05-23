import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import PReLU
from keras import Sequential
import keras_tuner

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
def build_model(hp):
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(350, 350, 3)),
        layers.Conv2D(12, 3, padding='same', activation=PReLU()),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation=PReLU()),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(20, activation=PReLU()),
        layers.Dense(10, activation=PReLU()),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [0.001, 0.0001]), beta_1=hp.Choice('beta_1', [0.8, 0.85, 0.9]), beta_2=hp.Choice('beta_2', [0.910, 0.950, 0.999]), epsilon=hp.Choice('epsilon', [1e0-5, 1e0-6, 1e-07]), decay = hp.Choice('decay', [0.0, 0.01, 0.05, 0.1]))
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
    return model

#Tuner to find the best hyperparameters
tuner = keras_tuner.tuners.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10)

tuner.search(train_ds, epochs=5, validation_data=val_ds)
best_model = tuner.get_best_models()[0]

optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay = weight_decay)

#model.summary()

#history = model.fit(
#    train_ds,
#    validation_data = val_ds,
#    epochs=epochs
#)