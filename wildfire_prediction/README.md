# 📡 Fire Prediction Project

This project aims to predict wildfires using deep learning techniques. It involves training a convolutional neural network (CNN) model on a dataset of wildfire images to classify them into two classes: "nowildfire" and "wildfire". The project utilizes the TensorFlow framework and Google Colab environment for development.

#### Dataset
The dataset consists of two directories: "train" and "valid". The "train" directory contains a large number of images that either had a fire or didn't have a fire, while the "valid" directory contains images for validation. The dataset is preprocessed using TensorFlow's `image_dataset_from_directory, where images are normalized. A subset of images is selected for training and validation. The dataset that I used can be found [here.](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)

#### Model Architecture
The CNN model is built using the Sequential API of Keras, a high-level API of TensorFlow. The model architecture consists of multiple layers, including convolutional layers, activation layers, pooling layers, and dense layers. Batch normalization and dropout techniques are employed to improve the model's performance and generalization ability. The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

#### Training and Evaluation
The model is trained using the `fit` method, where the training dataset is provided along with the validation dataset. The training is performed for a specified number of epochs, with the loss and accuracy metrics monitored during the training process.

#### Saving the Model
Once training is completed, the trained model is saved in the Hierarchical Data Format 5 (HDF5) file format with the filename "fire_model.h5". The saved model can be later loaded for prediction or deployment in the web application.

For the full project, including the web application implementation using Node.js, Express, and TensorFlow.js, please refer to the [complete project repository](https://github.com/snowjacob/FirePredictionJS).

Please note that this README provides a brief overview of the specific notebook file and its main functionalities. For more detailed information and a comprehensive understanding of the project, refer to the complete project documentation and code.
